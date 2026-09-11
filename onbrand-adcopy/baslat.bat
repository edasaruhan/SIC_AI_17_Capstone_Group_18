@echo off
title OnBrand AdCopy - Baslatici
color 0A

echo ================================================
echo        OnBrand AdCopy - Sistem Baslatici
echo ================================================
echo.

cd /d "%~dp0"

echo [1/4] Python kontrol ediliyor...
where python >nul 2>nul
if errorlevel 1 (
    echo HATA: Python bulunamadi! Lutfen Python 3.11+ kurun.
    pause
    exit /b 1
)

echo [2/4] API anahtari .env dosyasindan yukleniyor...
if not exist ".env" (
    echo UYARI: .env dosyasi bulunamadi!
    echo Lutfen .env.example dosyasini kopyalayip icine:
    echo   GOOGLE_API_KEY=bura-anahtarin
    echo satirini yazin.
    pause
    exit /b 1
)
for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    if /I "%%a"=="GOOGLE_API_KEY" set "GOOGLE_API_KEY=%%b"
)
if "%GOOGLE_API_KEY%"=="" (
    echo UYARI: .env icinde GOOGLE_API_KEY bos veya yok.
    pause
    exit /b 1
)

echo [3/4] Model kontrol ediliyor...
if not exist "models\xgboost_ctr_model.pkl" (
    echo Model bulunamadi, gercek veriyle egitiliyor...
    python scripts\train_main_model.py
    if errorlevel 1 (
        echo HATA: Model egitilemedi!
        pause
        exit /b 1
    )
) else (
    echo Model hazir.
)

echo [4/4] Streamlit uygulamasi baslatiliyor...
echo.
echo Tarayicinizda acilacak. Kapatmak icin bu pencereyi kapatin.
echo.
python -m streamlit run app/streamlit_app.py

pause