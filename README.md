+A Educação - Engenheiro de Inteligência Artificial
===================

[![N|Solid](https://maisaedu.com.br/hubfs/site-grupo-a/logo-mais-a-educacao.svg)](https://maisaedu.com.br/) 

O objetivo deste desafio é avaliar as competências técnicas dos candidatos a Engenheiro de Inteligência Artificial na Maior Plataforma de Educação do Brasil.

Neste teste prático, você será apresentado a um conjunto de dados simulado relacionado aos fundamentos de programação. Seu objetivo é realizar a indexação dos diferentes tipos de dados e desenvolver um prompt de sistema de aprendizagem adaptativa que gere conteúdos dinâmicos conforme as dificuldades e desconhecimentos do usuário sobre o tema.

# Sobre os dados

O conjunto de dados definidos para este teste é de propriedade intelectual da +A Educação e deverá ser utilizado exclusivamente para esta finalidade, sendo composto por:

- **Textos:** Uma coleção de textos extraídos de módulos de aprendizagem.
- **PDFs:** Livros e manuais introdutórios sobre o conteúdo abordado.  
- **Vídeos:** Pequenos vídeos de dicas de professor, que explicam o conteúdo abordado. Os vídeos estão em formatos .mp4 

O conjunto de dados pode ser obtido [clicando aqui](https://github.com/grupo-a/challenge-artificial-intelligence/tree/main/resources).

# Requisitos
## Etapa 1: Indexação dos Dados
Defina a ferramenta de indexação que considere adequada para o projeto. Recomenda-se escolher uma ferramenta que possa suportar a geração dinâmica de conteúdos adaptativos no prompt de IA generativa.
Indexe os diferentes tipos de dados para permitir uma busca eficiente e relevante.

Os textos devem ser indexados para permitir uma busca eficiente por palavras-chave e frases relevantes.

Os PDFs devem ser processados para extrair texto pesquisável e metadados importantes.

Os vídeos devem ser transcritos, se possível, e indexados com base no texto transcrito, juntamente com metadados descritivos.

As imagens devem ser indexadas considerando metadados relevantes, como tags, descrições e informações sobre o conteúdo visual.


## Etapa 2:  Prompt de Aprendizagem Adaptativa
Construa um prompt interativo, utilizando as tecnologias que julgar apropriadas, que identifique as dificuldades e lacunas de conhecimento dos usuários em um diálogo fluido e intuitivo para avaliar e entender as áreas onde seu conhecimento sobre um tema específico pode ser insuficiente. O escopo deve estar limitado ao conteúdo indexado.  Durante as interações, inclua mecanismos que permitam identificar as preferências dos usuários quanto ao formato de aprendizado mais efetivo para eles, seja texto, vídeo ou áudio, adaptando-se assim às suas preferências pessoais de consumo de conteúdo.

Baseado nas interações analise as dificuldades e gere conteúdos dinâmicos curtos em diferentes formatos (vídeos, áudios, textos) para abordar as necessidades específicas de aprendizagem do usuário. Os conteúdos devem ser relevantes, informativos e adaptados ao nível de conhecimento do usuário.

# Critérios de avaliação
- Qualidade de escrita do código.
- Organização do projeto.
- Lógica da solução implementada.
- Capacidade de escolher as tecnologias apropriadas para indexação de dados e desenvolvimento do sistema de aprendizagem adaptativa.
- Eficiência na manipulação de diferentes tipos de dados (textos, PDFs, vídeos).
- Competência na construção de prompts de IA que geram conteúdos adaptativos dinâmicos.
- Capacidade de integrar diferentes componentes do sistema (indexação, interface de usuário, geração de conteúdo) de forma coesa.

# Diferenciais
- Avaliação da capacidade do sistema em identificar corretamente as dificuldades dos usuários e adaptar o conteúdo de aprendizagem conforme necessário.
- Relevância, informatividade e adaptação do conteúdo gerado ao nível de conhecimento do usuário.
- Performance do sistema em diferentes condições de uso e sua capacidade de escalar conforme o aumento do número de usuários.

# Instruções de entrega
1. Crie um fork do repositório no seu GitHub
2. Faça o push do código desenvolvido no seu Github
3. Inclua um arquivo chamado COMMENTS.md explicando
    - Decisão da arquitetura utilizada
    - Lista de bibliotecas de terceiros utilizadas
    - O que você melhoraria se tivesse mais tempo
    - Quais requisitos obrigatórios que não foram entregues
4. Informe ao recrutador quando concluir o desafio junto com o link do repositório
5. Após revisão do projeto junto com a equipe de desenvolvimento deixe seu repositório privado
