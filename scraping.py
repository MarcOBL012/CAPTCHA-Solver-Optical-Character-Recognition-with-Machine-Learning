from selenium import webdriver
from selenium.webdriver.common.by import By
import base64
# Configuración de Selenium
options = webdriver.ChromeOptions()
options.add_argument("--headless") # Ejecuta Chrome sin GUI
driver = webdriver.Chrome(options=options)

# URL del CAPTCHA
captcha_url = "https://apps2.mef.gob.pe/consulta-vfp-webapp/consultaExpediente.jspx"

for i in range(40):
    # Navegar a la URL del CAPTCHA
    driver.get(captcha_url)
    captchaImage = driver.find_element(By.XPATH,'//*[@id="captchaImage"]')
    captchaImageSave = driver.execute_async_script("""
                    var ele = arguments[0], callback = arguments[1];
                    ele.addEventListener('load', function fn(){
                      ele.removeEventListener('load', fn, false);
                      var cnv = document.createElement('canvas');
                      cnv.width = this.width; cnv.height = this.height;
                      cnv.getContext('2d').drawImage(this, 0, 0);
                      callback(cnv.toDataURL('image/jpeg').substring(22));
                    }, false);
                    ele.dispatchEvent(new Event('load'));
                    """, captchaImage)
    # Guardar la imagen en el disco
    filename = ('00000' + str(i))[-5:] + '.png'
    with open(filename, 'wb') as f:
        f.write(base64.b64decode(captchaImageSave))
    print(filename)

# Cerrar el navegador
driver.quit()

print('end')