# High-Performance C++ Linear Algebra & Neural Network Engine

Este motor es una implementación desde cero (from scratch) de una biblioteca de álgebra lineal optimizada y un framework de Deep Learning modular desarrollado en C++17. El proyecto ha sido diseñado priorizando la eficiencia computacional mediante paralelismo masivo y una arquitectura de grafos de operación.

## 1. Arquitectura del Sistema

El framework se divide en dos grandes núcleos interdependientes:

### A. Core Matemático (Math)

Ubicado en `src/math/`, gestiona la abstracción de tensores de rango 2 (Matrices).

* **Gestión de Memoria:** Implementa un esquema de almacenamiento contiguo para maximizar el cache locality.

* **Aceleración de Hardware:** Utiliza directivas OpenMP (SIMD y Parallel For) para la vectorización de operaciones elementales y multiplicación de matrices.

* **Broadcasting Engine:** Capacidad nativa para operar matrices de distintas dimensiones compatibles (ej. suma de bias vectorizada sobre batches).

* **Polimorfismo de Datos:** Basado íntegramente en templates para soportar precisión simple (float), doble (double) o tipos personalizados.

### B. Motor de Redes Neuronales (NN)

Ubicado en `src/nn/`, sigue un diseño orientado a objetos para el cálculo de gradientes.

* **Abstracción de Operaciones:** Cada capa se descompone en un grafo de Operations, facilitando la implementación de la retropropagación (Backpropagation) automática.

* **Initialization Suite:** Implementación de inicialización de pesos Xavier/Glorot Normal para prevenir el desvanecimiento del gradiente.

* **Optimización Avanzada:** Soporte para descenso de gradiente estocástico (SGD) y el algoritmo Adam (Adaptive Moment Estimation) con corrección de sesgo.

## 2. Componentes Principales

### Modelos y Capas

* `NN::Model`: Orquestador principal. Gestiona el ciclo de vida del entrenamiento, métricas y callbacks.

* `NN::Layer::Sequential`: Contenedor de capas que automatiza el flujo de tensores (forward/backward).

* `NN::Layer::Dense`: Capa totalmente conectada (Fully Connected) con soporte para activaciones integradas.

### Funciones de Activación y Coste

* **Activaciones**: ReLU, Sigmoid y Tanh (optimizadas para gradientes locales).

* **Coste**: MSE (Mean Squared Error) y Cross-Entropy (implícito/extensible).

### Optimización

* **Adam Optimizer**: Implementa momentos de primer y segundo orden para una convergencia acelerada.

* **Hyperparameter Control**: Configuración fina de $\beta_1$, $\beta_2$ y $\epsilon$.

## 3. Guía de Uso Rápido

Definición de un Perceptrón Multicapa (MLP)

El API ha sido diseñado para ser intuitivo y similar a frameworks modernos como Keras o PyTorch.

```cpp
#include "nn/model.h"
#include "nn/layers.h"
#include "nn/activation_func.h"

// 1. Instanciar el contenedor secuencial
auto layers = std::make_shared<NN::Layer::Sequential<float>>();

// 2. Construir la arquitectura
layers->add(std::make_shared<NN::Layer::Dense<float>>(
    128, std::make_shared<NN::ActFunc::ReLU<float>>())
);
layers->add(std::make_shared<NN::Layer::Dense<float>>(
    10, std::make_shared<NN::ActFunc::Sigmoid<float>>())
);

// 3. Configurar el modelo
NN::Model<float> model;
model.set_layers(layers);

// 4. Compilación (Inyección de pérdida y optimizador)
auto optimizer = std::make_shared<NN::Optimizer::Adam<float>>(0.001f);
auto loss = std::make_shared<NN::CostFunc::MSE<float>>();
model.compile(loss, optimizer);

// 5. Entrenamiento
model.fit(x_train, y_train, 50 /*epochs*/, 10 /*verbose*/);
```

### Visualización de la Arquitectura

El método model.summary() genera un reporte detallado en consola:

Layer (type)             Output Shape             Param #        
=================================================================
Dense_1                  (None, 128)              8192           
Dense_2                  (None, 10)               1280           
=================================================================
Total params: 9472
Trainable params: 9472


## 4. Requisitos Técnicos e Instalación

### Prerrequisitos

* **Compilador**: GCC 7+ o Clang con soporte para C++17.

* **Paralelismo**: Soporte para libgomp (OpenMP).

* **Sistema de Construcción**: CMake 3.10+.

### Compilación

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 5. Rendimiento y Optimización

Para obtener el máximo rendimiento de la librería de matrices, asegúrese de compilar con las banderas de optimización de arquitectura:

* `-O3`: Optimización de nivel máximo.
* `-fopenmp`: Habilita el paralelismo multinúcleo.
* `-march=native`: Permite el uso de instrucciones vectoriales AVX/AVX2/AVX-512 según el procesador.
