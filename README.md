# 🐸 peppe - Fine-tuning GPT-2 com Greentexts do 4chan

Este projeto realiza o fine-tuning do modelo GPT-2 utilizando um dataset de greentexts coletados do 4chan. 

## 📖 Sobre o Projeto

O **peppe** é um experimento de aprendizado de máquina que treina o modelo de linguagem GPT-2 da OpenAI para gerar textos no estilo característico das "greentexts" — histórias curtas e humorísticas originadas nos fóruns do 4chan, tipicamente escritas em linhas que começam com `>`.

## 🎯 Objetivo

O objetivo principal é fazer com que o modelo aprenda o estilo único de escrita das greentexts, incluindo:
- Formato de texto com linhas iniciando com `>`
- Narrativa em primeira pessoa
- Tom humorístico e absurdo
- Estrutura típica de "história de anônimo"

## 📦 Dependências

O projeto utiliza as seguintes bibliotecas principais:

- **transformers** (>=4.57.3) - Para carregar e treinar o modelo GPT-2
- **torch** (>=2. 9.1) - Framework de deep learning
- **datasets** (>=4. 4.1) - Para manipulação do dataset
- **tiktoken** (>=0.12.0) - Tokenização
- **tqdm** (>=4.67.1) - Barras de progresso

## 🚀 Instalação

```bash
# Clone o repositório
git clone https://github. com/mnsgrosa/peppe.git
cd peppe

# Instale as dependências usando uv
uv sync

# Ou usando pip
pip install -e .
```

## 📂 Estrutura do Projeto

```
peppe/
├── main.py              # Script principal
├── src/                 # Código fonte do projeto
├── greentext_data/      # Dataset de greentexts
├── log/                 # Logs de treinamento
├── pyproject.toml       # Configurações do projeto
└── README.md            # Este arquivo
```

## 🗃️ Dataset

O dataset utilizado consiste em greentexts coletadas do 4chan.  Greentexts são um formato de postagem característico dos imageboards, onde as linhas começam com o símbolo `>` (que aparece em verde no site original, daí o nome). 

### Características do Dataset:
- Formato de texto único e reconhecível
- Histórias curtas e narrativas
- Conteúdo humorístico e satírico
- Linguagem informal da internet

## 🧠 Sobre o GPT-2

O GPT-2 (Generative Pre-trained Transformer 2) é um modelo de linguagem desenvolvido pela OpenAI. Através do processo de fine-tuning, adaptamos o modelo pré-treinado para gerar textos específicos no estilo greentext.

### Processo de Treinamento:
1. Carregamento do modelo GPT-2 pré-treinado
2. Preparação e tokenização do dataset de greentexts
3. Fine-tuning do modelo com os dados específicos
4.  Avaliação e geração de novos textos

## 📝 Uso

Para rodar o projeto e necessario criar o dataset dos greentexts do 4chan, para treinar o modelo.


1) Download do dataset

```python
uv run src/model/greentexts.py
```

2) Treinamento do modelo

```python
uv run src/model/train.py
```

3) Ataque ao modelo

O modelo por ser um gerador de texto podemos atacar gerando dados ofensivos ou preconceituosos, para isso basta passar um prompt simples que o modelo ira completar o texto.

```python
uv run src/model/attack.py
```

Isso ira gerar o txt com o prompt base em um txt

4) Caso queira testar o modelo por conta propria basta usar o seguinte trecho de codigo

```python
import torch
import tiktoken
from gpt2 import GPT, GPTConfig
from torch.nn import functional as F

model = GPT(
    GPTConfig(block_size=256, vocab_size=50304, n_layer=4, n_head=4, n_embd=256)
)

enc = tiktoken.get_encoding("gpt2")

weights_path = "./weights/gpt2_weights.pth"
model.load_state_dict(torch.load(weights_path))

enc_input = torch.tensor(enc.encode("Seu prompt aqui"), dtype=torch.long)
enc_input = enc_input.unsqueeze(0).repeat(4, 1)

model.eval()
with torch.no_grad():
    with torch.autocast(device_type = "cpu", dtype = torch.bfloat16):
        logits, loss = model(input)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    generator = torch.Generator().manual_seed(42)
    ix = torch.multinomial(topk_probs, 1, generator=generator)
    xcol = torch.gather(topk_indices, -1, ix)
    text = torch.cat((text, xcol), dim=1)
for i in range(4):
    tokens = text[i, :32].tolist()
    decoded = enc.decode(tokens)

with open("output.txt", "a") as f:
    f.write(f"Output {i+1}:\n{decoded}\n\n")
```

## ⚠️ Aviso

Este projeto é puramente educacional e experimental. O conteúdo gerado pelo modelo pode refletir o estilo e tom do dataset de treinamento.  Use com responsabilidade. 

## 📄 Licença

Este projeto é de código aberto.  Sinta-se livre para usar, modificar e distribuir. 

## 🤝 Contribuições

Contribuições são bem-vindas!  Sinta-se à vontade para abrir issues ou pull requests.

---

*Feito com 🐸 e muito fine-tuning*
