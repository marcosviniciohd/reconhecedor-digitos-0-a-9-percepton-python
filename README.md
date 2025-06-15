# Reconhecedor de Dígitos 0-9 com Perceptron (Python)

Este projeto implementa um reconhecedor de dígitos (0 a 9) utilizando o algoritmo Perceptron Multiclasse (um perceptron para cada dígito) em Python.

## Sobre o Algoritmo

- **Entrada:** Cada dígito é representado por uma matriz 5x3 (15 pixels) + 1 bias, totalizando 16 entradas.
- **Saída:** O sistema possui 10 perceptrons, cada um treinado para reconhecer um dígito (0 a 9) usando abordagem one-vs-all.
- **Treinamento:** O treinamento ajusta os pesos de cada perceptron para que ele ative apenas para o seu dígito correspondente.
- **Reconhecimento:** O dígito é reconhecido pelo perceptron que apresentar a maior ativação para a entrada fornecida.

## Como Executar (Python)

1. Certifique-se de ter o Python instalado ([https://www.python.org/downloads/](https://www.python.org/downloads/)).
2. Instale o pacote numpy:
   ```bash
   pip install numpy
   ```
3. Execute o script:
   ```bash
   python perceptron_digitos.py
   ```
4. O script irá treinar o perceptron e mostrar no terminal se ele reconhece corretamente os dígitos de 0 a 9.

## Resultados Esperados
```
Épocas de treinamento: 25
Esperado: 0, Predito: 0
Esperado: 1, Predito: 1
...
Esperado: 9, Predito: 9
Acurácia: 10/10
```

## Observações
- O reconhecimento depende dos padrões de entrada definidos no código.
- Para melhorar a robustez, adicione mais exemplos de cada dígito.
- O código pode ser expandido para reconhecer dígitos desenhados à mão com pré-processamento adequado.

---
