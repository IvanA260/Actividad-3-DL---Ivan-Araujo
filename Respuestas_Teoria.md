# Respuestas Teoría: Introducción al Deep Learning (Bloque 1 y 2)

## Bloque 1: Visión General

### P1. Mapa rápido de Deep Learning
**Deep Learning (DL)** es una rama del Machine Learning que aprende representaciones en capas para transformar datos crudos en predicciones. Nace conceptualmente con el **perceptrón**, despega con la **backpropagation** y hoy impulsa la era de la IA generativa en visión y lenguaje. Su principal ventaja frente al ML clásico es que reduce la necesidad de diseñar "features a mano", aprendiéndolas automáticamente. Sus límites actuales son la gran necesidad de datos, cómputo y el riesgo de sobreajuste.
* **Ejemplo:** Clasificación de fotos de gatos vs. perros.
* **Duda:** ¿En qué casos específicos un modelo más simple supera a un modelo de Deep Learning debido a la falta de datos?

### P2. Del ML al DL en una imagen mental
El ML clásico usa características ("features") diseñadas manualmente, mientras que el DL las aprende automáticamente capa a capa, donde cada capa recodifica la información de la anterior.
**Diagrama Mental:**
* **Datos crudos:** (Entrada de audio o imagen).
* **Capa 1:** Aprende patrones simples (bordes o frecuencias).
* **Capa 2:** Aprende estructuras más complejas (formas o fonemas).
* **Salida:** Concepto final (palabra u objeto).

### P3. MLP en 60 segundos
Un **MLP (Perceptrón Multicapa)** tiene capas densas compuestas por pesos ($W$) y sesgos ($b$).
* **Forward pass:** Propaga las activaciones desde la entrada hacia la salida.
* **Backprop:** Ajusta los pesos calculando gradientes desde el error hacia atrás.
* **Conclusión:** Arquitecturas como CNNs o Transformers se entienden como extensiones que añaden estructura espacial o atencional, pero mantienen la base de optimización por gradiente.

### P4. Ejemplo numérico guiado
Simulación de forward pass en red 2-3-1 con números sencillos:
1. **Entrada:** Valores iniciales $x$.
2. **Cálculo ($z$):** Se realiza la suma ponderada de entradas por pesos más el sesgo ($z = W \cdot x + b$).
3. **Activación:** Se aplica la función no lineal (ej. ReLU) sobre $z$.
4. **Salida:** El valor final resultante tras la última capa.
* **Interpretación:** Este valor numérico representa la predicción cruda; si es clasificación, se puede interpretar como la confianza o probabilidad de pertenecer a una clase.

### P5. Por qué importa la pérdida
Elijo la **Entropía Cruzada (Cross-Entropy)**. Esta función mide el desajuste probabilístico entre la predicción y la etiqueta real; penaliza fuertemente los errores cometidos con alta confianza. Minimizarla es crucial porque guía el ajuste de pesos: el objetivo del entrenamiento es reducir este valor de pérdida.

---

## Bloque 2: Temas Relevantes

### P6. Activaciones sin dolor
* **Sigmoide/Tanh:** "Comprimen" la salida en rangos fijos ((0,1) o (-1,1)). Problema: saturación de gradientes.
* **ReLU:** Acelera el cálculo y evita saturación positiva. Problema: neuronas muertas si $z < 0$ persistentemente.
* **Elección:** Usaría **ReLU** en capas ocultas porque es eficiente y reduce el problema del desvanecimiento del gradiente.

### P7. Arquitectura MLP mínima
**Esquema:**
* **Entradas:** $n$ neuronas.
* **Capa Oculta:** $h$ neuronas. Definida por matriz de pesos $W^{(1)}$ y vector de sesgos $b^{(1)}$.
* **Salidas:** $m$ neuronas. Definida por $W^{(2)}$ y $b^{(2)}$.
Las dimensiones de las matrices de pesos deben casar exactamente con las capas anteriores y posteriores.

### P8. Forward pass con ecuaciones
Las ecuaciones clave son:
$$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma(z^{(l)})$$
El flujo va de izquierda a derecha. Si cambias la función $\sigma$, cambias el rango de activación y la no-linealidad que la red es capaz de aprender.

### P9. Backprop sin miedo
$\delta^{(l)}$ es el error local en una capa específica. Mide cuánto contribuye esa neurona al error total. Nos ayuda a saber cuánto cambiar cada peso porque el gradiente final depende directamente de este valor local; si el error local es alto, el ajuste del peso será mayor.

### P10. Pérdida y decisión
* **MSE:** Mide el error cuadrático medio; usar en **regresión** (salida continua).
* **Cross-Entropy:** Mide el error de clasificación; usar en problemas **multi-clase** o binarios con probabilidad.

### P11. Ejercicio numérico corto
**Caso:** $\delta^{(2)}$ es negativo y la entrada $a^{(1)}$ es grande (positiva).
**Razonamiento:** El gradiente es proporcional a $\delta \cdot a$, por lo que el gradiente es negativo. La regla de actualización es $W \leftarrow W - \eta \cdot (\text{gradiente})$. Restar un valor negativo equivale a sumar.
**Resultado:** El peso **sube** (aumenta).

### P12. Historia express
1. **1958 (Perceptrón):** Nace la neurona artificial simple.
2. **1986 (Backprop):** Se establece el método para entrenar capas ocultas.
3. **2012 (Era DL):** Convergencia de GPUs, grandes datos (ImageNet) y algoritmos profundos a escala.

### P13. Limitaciones del MLP
Los MLPs densos fallan en visión porque:
1. Ignoran la **estructura espacial** (aplanan la imagen).
2. Tienen demasiados parámetros al no compartir pesos.
* **Solución CNN:** Usan convolución y localización para explotar la estructura espacial eficientemente.

### P14. Métrica de éxito
Si `train` sube y `valid` baja, el problema es **Sobreajuste (Overfitting)**.
**2 Remedios:**
1. **Regularizar** (ej. Dropout).
2. Conseguir **más datos** o aplicar Data Augmentation.

### P15. Mini-resumen personal
* **Entendido:** El DL automatiza la extracción de características mediante capas jerárquicas y backpropagation.
* **Dudas:** Me falta práctica intuitiva sobre cómo ajustar el Learning Rate correctamente.
* **Aplicación:** Aplicaría DL para reconocimiento de voz en entornos ruidosos (audio a texto).