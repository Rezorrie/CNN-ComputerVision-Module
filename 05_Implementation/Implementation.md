# IMPLEMENTATION

## Membuat VENV (Virtual Environment)
```bash
# Buat Venv
python3 -m venv myenv # myenv bisa diganti dengan nama venv lainnya

# Aktivasi
source myenv/bin/activate     # Linux / MacOS
myenv\Scripts\activate.bat    # CMD
.\myenv\Scripts\Activate.ps1  # PowerShell

# Install VEnv kernel apabila diperlukan
pip install ipykernel
python -m ipykernel install --user --name=myenv
```
