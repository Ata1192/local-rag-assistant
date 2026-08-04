@echo off
echo =======================================================
echo     LOCAL RAG ASSISTANT (Yapaz Zeka Uygulamasi)
echo =======================================================
echo.
echo Sanal ortam (venv) aktif ediliyor...
call venv\Scripts\activate.bat

echo.
echo Analiz ve Optimizasyon ayarlari yapiliyor...
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo.
echo Uygulama baslatiliyor, lutfen bekleyin...
echo Tarayiciniz otomatik olarak acilacaktir.
echo.
echo Kapatmak icin bu pencereyi (X) ile kapatabilirsiniz.
echo.
streamlit run app.py
pause
