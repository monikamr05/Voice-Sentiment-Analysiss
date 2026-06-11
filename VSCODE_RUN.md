# Run in VS Code (Fix "Install TensorFlow")

## Step 1 — Open the correct folder
Open **`voice_sentiment_analysis`** (not only the parent `sentiment` folder).

Or open: **`sentiment.code-workspace`** (in the parent folder).

## Step 2 — Select Python interpreter
1. Press **Ctrl + Shift + P**
2. Type: **Python: Select Interpreter**
3. Choose: **`.venv (Python 3.12.x)`**
   - Path must end with: `voice_sentiment_analysis\.venv\Scripts\python.exe`
4. **Do NOT** use Python 3.14 — TensorFlow does not work on 3.14.

## Step 3 — Run the app
- Press **F5** → pick **"Run Flask App"**
- Or Terminal:
  ```powershell
  cd voice_sentiment_analysis
  .\run.ps1
  ```

## Step 4 — If VS Code still shows "Install TensorFlow"
That is only a yellow warning. Ignore it, or reload window:
**Ctrl + Shift + P** → **Developer: Reload Window**

## Verify TensorFlow
```powershell
.\.venv\Scripts\python -c "import tensorflow; print('OK')"
```

If you see **Application Control policy blocked**:
- Windows Security → Virus & threat protection → Manage settings
- Add exclusion for folder: `voice_sentiment_analysis`
- Or run PowerShell **as Administrator** once and run `.\run.ps1`

## Default logins
- Admin: `admin` / `admin123`
- User: register at `/register`
