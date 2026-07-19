from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Datos de entrenamiento (Ejemplos para que la IA aprenda)
mensajes = [
    "Oferta ganar dinero facil click aqui",
    "Hola como estas te veo mañana",
    "Premio urgente reclama tus monedas gratis",
    "Necesito el reporte de machine learning para hoy",
    "Descuento exclusivo compra ya"
]
# 0 = Mensaje Normal, 1 = Spam
etiquetas = [1, 0, 1, 0, 1]

# 2. Convertir el texto a números que el modelo entienda
vectorizador = CountVectorizer()
X_entrenamiento = vectorizador.fit_transform(mensajes)

# 3. Crear y entrenar el modelo de Machine Learning
modelo = MultinomialNB()
modelo.fit(X_entrenamiento, etiquetas)

# 4. Probar el modelo con un mensaje nuevo
nuevo_mensaje = ["Ganar dinero gratis ya"]
X_nuevo = vectorizador.transform(nuevo_mensaje)
prediccion = modelo.predict(X_nuevo)

# 5. Mostrar el resultado en pantalla
if prediccion[0] == 1:
    print(f"El mensaje: '{nuevo_mensaje[0]}' es SPAM 🚨")
else:
    print(f"El mensaje: '{nuevo_mensaje[0]}' es NORMAL ✅")
