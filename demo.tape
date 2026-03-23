# safha demo
Output demo.gif
Set Shell "zsh"
Set FontSize 13
Set Width 1200
Set Height 700
Set Theme "Catppuccin Mocha"
Set TypingSpeed 25ms
Set Padding 16

Hide
Type "source .venv/bin/activate && clear"
Enter
Sleep 500ms
Show

Sleep 300ms
Type 'safha scrape "https://ar.wikipedia.org/wiki/ذكاء_اصطناعي" --output /tmp/demo.jsonl'
Sleep 300ms
Enter
Sleep 5s

Type "safha detect /tmp/demo.jsonl"
Sleep 300ms
Enter
Sleep 3s

Type "safha stats /tmp/demo.jsonl"
Sleep 300ms
Enter
Sleep 4s

Sleep 2s
