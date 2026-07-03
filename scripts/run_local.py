#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess
import sys
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHALLENGES_ROOT = REPO_ROOT / "challenges"
for p in [str(REPO_ROOT), str(CHALLENGES_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import ctypes
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Как это работает (ctypes + .so):
#
#  1. nvcc компилирует starter.cu в .so (shared library) — ELF-файл, в котором
#     лежит скомпилированный машинный код функции solve(), готовый к запуску.
#     Флаг --shared говорит линкеру собрать динамическую библиотеку,
#     а -Xcompiler -fPIC — что код можно загружать по любому адресу (PIC).
#
#  2. ctypes.CDLL("...") загружает .so в адресное пространство текущего Python-
#     процесса через системный вызов dlopen(). Теперь Python может вызывать
#     экспортированные C-функции так, как если бы они были обычными функциями.
#
#  3. argtypes и restype — это описание протокола вызова (ABI). Они говорят
#     ctypes, как конвертировать Python-объекты в C-типы (marshalling):
#       - ctypes.c_float       → 4-байтовое число float (регистр или стек)
#       - ctypes.c_size_t      → 8-байтовое беззнаковое целое (типа size_t)
#       - ctypes.POINTER(...)  → 8-байтовый указатель (void* / device ptr)
#     Без argtypes ctypes пытается угадать тип по Python-значению, что
#     на 64-битных системах приводит к обрезанию указателей до 32 бит.
#
#  4. GPU-указатели: PyTorch тензоры хранят данные на GPU. Мы не копируем
#     данные в CPU — вместо этого берём их device-адрес через data_ptr()
#     и передаём его в solve() как ctypes-указатель. Функция solve() (CUDA)
#     работает с этим адресом напрямую через операторы разыменования *ptr
#     или gpu_arrays[idx]. Весь ввод-вывод происходит на GPU, без копирования
#     через PCIe.
#
#  Схема вызова:
#    Python                         C/CUDA (.so)
#    ──────                         ───────────
#    tensor.data_ptr() ──→ getCudaPtr() ──→ device_address (uint64)
#         │                                      │
#         ▼                                      ▼
#    ctypes.cast(addr, POINTER(c_float)) ──→ float* ptr (в аргументе solve)
#         │                                      │
#         ▼                                      ▼
#    solve_func(ptr_A, ptr_B, ptr_C, N) ──→ solve(float* A, float* B, float* C, int N)
#                                              │
#                                              ▼
#                                         kernel<<<...>>>(A, B, C, N)
#                                              │
#                                              ▼
#         ◄── GPU память изменена ────    tensor_C заполнен результатом
#         torch.allclose(…, expected)      CUDA kernel записал данные в буфер C
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def find_solution_file(challenge_dir: Path) -> Path:
    for folder in ["starter"]:
        for name in ["starter.cu"]:
            path = challenge_dir / folder / name
            if path.exists():
                return path
    raise FileNotFoundError(f"Не найден .cu файл в starter/ или solution/ внутри {challenge_dir}")

def get_cuda_ptr(tensor: torch.Tensor):
    # data_ptr() возвращает сырой адрес GPU-памяти, где лежит тензор (int).
    # ctypes.cast() превращает этот int в типизированный ctypes-указатель
    # (LP_c_float = pointer to float), который можно передать в C-функцию.
    return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))

def run_ncu_profile(task, so_file: Path, signature: dict, arg_names: list, ncu_flags: str) -> int:
    """Запускает ncu профилирование на performance-тесте через временный Python-скрипт."""
    import tempfile
    import shutil

    perf_case = task.generate_performance_test()
    challenge_name = task.name.lower().replace(" ", "_")

    # Проверка ncu
    if not shutil.which("ncu"):
        logger.error("ncu не найден в PATH. Установите NVIDIA Nsight Compute.")
        return 1

    # Сохраняем тензоры в .pt файл, чтобы не раздувать wrapper гигантскими literal'ами
    pt_path = so_file.with_suffix(".profile_data.pt")
    torch.save({name: val for name, val in perf_case.items() if isinstance(val, torch.Tensor)}, pt_path)

    # Собираем аргументы для вызова — передаём ctypes-инстансы, чтобы
    # ctypes корректно маршалил их без необходимости в argtypes
    setup_lines = []
    call_args = []
    for name in arg_names:
        val = perf_case[name]
        ctype = signature[name][0]
        if isinstance(val, torch.Tensor):
            setup_lines.append(
                f"_{name}_ptr = ctypes.cast(data[\"{name}\"].data_ptr(), ctypes.POINTER(ctypes.c_float))"
            )
            call_args.append(f"_{name}_ptr")
        else:
            # Скаляр — оборачиваем в его ctypes-тип (c_size_t, c_int, ...)
            setup_lines.append(f"_{name}_val = ctypes.{ctype.__name__}({val!r})")
            call_args.append(f"_{name}_val")

    wrapper = f"""#!/usr/bin/env python3
import ctypes
import torch

data = torch.load("{pt_path}", map_location="cuda", weights_only=True)

so = ctypes.CDLL("{so_file}")
solve = so.solve
solve.restype = None

{chr(10).join(setup_lines)}

torch.cuda.synchronize()
solve({', '.join(call_args)})
torch.cuda.synchronize()
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"ncu_{challenge_name}_", delete=False
    ) as f:
        wrapper_path = f.name
        f.write(wrapper)

    output_path = f"profile_{challenge_name}"
    logger.info("🚀 Запуск NCU профилирования...")
    logger.info("   ncu %s -o %s python %s", ncu_flags, output_path, wrapper_path)
    print()

    cmd = ["ncu"] + ncu_flags.split() + ["-o", output_path, "python", wrapper_path]
    result = subprocess.run(cmd)

    # Чистим временные файлы
    Path(wrapper_path).unlink(missing_ok=True)
    pt_path.unlink(missing_ok=True)

    if result.returncode == 0:
        logger.info("✅ Профиль сохранён: %s.ncu-rep", output_path)
        logger.info("   Открыть: ncu-ui %s.ncu-rep", output_path)
    else:
        logger.error("❌ NCU завершился с ошибкой (код %d)", result.returncode)

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Локальный запуск тестов для задач LeetGPU.")
    parser.add_argument("challenge_path", type=Path, help="Путь к папке с задачей")
    parser.add_argument("--ncu", action="store_true", help="Запустить профилирование NCU на performance-тесте")
    parser.add_argument("--ncu-flags", type=str, default="--set full", help="Флаги для ncu (по умолчанию: --set full)")
    args = parser.parse_args()

    challenge_dir = args.challenge_path.resolve()
    challenge_py = challenge_dir / "challenge.py"
    
    if not challenge_py.exists():
        logger.error("Файл challenge.py не найден в %s", challenge_dir)
        return 1

    # ── 1. Компиляция: .cu → .so ─────────────────────────────────────────
    # --shared      → динамическая библиотека (не исполнимый файл)
    # -Xcompiler -fPIC → position-independent code — чтобы .so можно было
    #                     загрузить в любой процесс через dlopen()
    # На выходе: starter.so — ELF-файл, содержащий скомпилированную solve()
    try:
        cu_file = find_solution_file(challenge_dir)
        so_file = cu_file.with_suffix(".so")
        
        logger.info("🛠️  Компиляция %s в общую библиотеку (.so)...", cu_file.name)
        subprocess.run([
            "nvcc", "-O3", "--shared", "-Xcompiler", "-fPIC",
            str(cu_file), "-o", str(so_file)
        ], check=True)
    except Exception as e:
        logger.error("❌ Ошибка компиляции nvcc: %s", e)
        return 1

    try:
        # 2. Инициализация изолированного модуля с доступом к глобальному контексту репозитория
        spec = importlib.util.spec_from_file_location("challenge_mod", str(challenge_py))
        mod = importlib.util.module_from_spec(spec)
        
        # Добавляем модуль в системный кэш, чтобы внутренние абсолютные импорты 'from core.xxx' сработали
        sys.modules["challenge_mod"] = mod
        
        spec.loader.exec_module(mod)
        task = mod.Challenge()
        logger.info("Загружена задача: %s", task.name)

        # ── 2. Загрузка .so и описание ABI через ctypes ────────────────
        # CDLL() вызывает dlopen() — .so загружается в наш процесс.
        # Теперь solve_func — это callable, который ведёт себя как C-функция.
        cuda_lib = ctypes.CDLL(str(so_file))
        solve_func = cuda_lib.solve
        
        # get_solve_signature() возвращает словарь:
        #   {"A": (ctypes.POINTER(ctypes.c_float), "in"),
        #    "B": (ctypes.POINTER(ctypes.c_float), "in"),
        #    "C": (ctypes.POINTER(ctypes.c_float), "out"),
        #    "N": (ctypes.c_size_t, "in")}
        #
        # argtypes = список ctypes-типов — описывает прототип C-функции.
        # Без этого ctypes не знает размер каждого аргумента и передаёт
        # всё как int (32 бита), что ломает 64-битные указатели на GPU.
        signature = task.get_solve_signature()
        arg_names = list(signature.keys())
        
        solve_func.argtypes = [signature[name][0] for name in arg_names]
        # restype = None — функция ничего не возвращает (void)
        solve_func.restype = None

        # 4. NCU профилирование на performance-тесте
        if args.ncu:
            return run_ncu_profile(task, so_file, signature, arg_names, args.ncu_flags)

        # 5. Прогон функциональных тестов
        test_cases = task.generate_functional_test()
        logger.info("Запуск %d функциональных тестов...", len(test_cases))
        
        passed = 0
        for idx, case in enumerate(test_cases):
            # Дублируем аргументы для безопасного вычисления эталона на PyTorch
            ref_args = {name: (case[name].clone() if isinstance(case[name], torch.Tensor) else case[name]) for name in arg_names}
            task.reference_impl(**ref_args)
            
            # Находим имя выходного тензора
            out_name = next(name for name, (_, direction) in signature.items() if direction == "out")
            expected_output = ref_args[out_name]
            
            # Обнуляем вашу матрицу/вектор вывода перед тестом
            case[out_name].zero_()

            # ── 3. Превращаем PyTorch-тензоры в C-указатели ─────────────
            # Тензоры уже лежат на GPU (case — из generate_functional_test).
            # Их data_ptr() — это адрес в VRAM, который нужно передать solve().
            # ctypes.cast() оборачивает int-адрес в LP_c_float (float*).
            # Скалярные аргументы (c_size_t, c_int) передаём как есть —
            # ctypes сам сконвертирует Python int в нужный C-тип.
            args_to_pass = []
            for name in arg_names:
                val = case[name]
                if isinstance(val, torch.Tensor):
                    args_to_pass.append(get_cuda_ptr(val))
                else:
                    args_to_pass.append(val)

            # ── 4. Вызов CUDA-функции через .so ──────────────────────────
            # solve_func(ptr_A, ptr_B, ptr_C, N) спускается в compiled-code
            # внутри .so, который запускает CUDA kernel на GPU.
            # Никаких копирований CPU↔GPU — все данные уже на девайсе.
            solve_func(*args_to_pass)
            # Синхронизация: ждём, пока GPU закончит kernel,
            # чтобы результат был виден в тензоре case[out_name].
            torch.cuda.synchronize()

            # Сверяем результат с эталоном по допускам задачи
            your_output = case[out_name]
            if torch.allclose(your_output, expected_output, atol=task.atol, rtol=task.rtol):
                passed += 1
            else:
                logger.error("❌ Тест %d провален!", idx + 1)
                max_diff = (your_output - expected_output).abs().max().item()
                logger.error("   Максимальное отклонение: %f (Допустимо: %f)", max_diff, task.atol)
                return 1

        logger.info("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! (%d/%d)", passed, len(test_cases))
        return 0

    except Exception as e:
        logger.error("⚠️  Ошибка во время выполнения тестов: %s", e)
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Убираем за собой бинарники
        if 'so_file' in locals() and so_file.exists():
            so_file.unlink()

if __name__ == "__main__":
    sys.exit(main())