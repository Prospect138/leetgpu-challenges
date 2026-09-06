import os
header = '''
<head>
  <script async="" id="MathJax-script" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
'''

ext = '.html'
root = os.path.abspath('..')
print(root)

for dirpath, dirname, filenames in os.walk(root):
    for file in filenames:
        if os.path.splitext(file)[1].lower() == ext:
            file_uri = os.path.join(dirpath, file)
            with open(file_uri, 'r+') as f:
                content = f.read()
                if header not in content:
                    f.seek(0, 0)
                    f.write(header + content)