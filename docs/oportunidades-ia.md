# Oportunidades de Produtos com IA Conectados ao D-Guard e Situator

Análise baseada nos manuais técnicos do **Seventh D-Guard (v7.1.4.9 | Web 1.46.0)** e **Seventh Situator (v5)**.

---

## 1. Filtro Inteligente de Falsos Positivos (AI Video Verification)
- **Problema real documentado:** O próprio manual do D-Guard (seção *5.2.10 Analítico*) alerta que os analíticos tradicionais sofrem com variação de iluminação e alta movimentação, gerando alarmes falsos e sobrecarga na fila do Situator.
- **Como conecta:**
  1. D-Guard envia snapshot (via **FTP** ou **HTTP GET**) ou stream (**RTSP porta 554**) no momento do evento.
  2. Um modelo VLM / detecção moderna (YOLO/VLM) valida se há realmente intrusão humana, veículo, arma, queda ou apenas chuva/sombra/animal.
  3. Só se confirmado, o serviço envia a ocorrência para a **API HTTP/JSON (ou Webhook) do Situator**, já com prioridade ajustada (*Crítico / Alto / Médio / Baixo*) e descrição enriquecida.
- **Modelo de negócio:** SaaS cobrado por câmera/mês para centrais de monitoramento (*D-Guard Center* / *Situator Center*).

## 2. Copiloto de Atendimento & Triagem Automática no Situator
- **Problema real documentado:** No Situator, o operador precisa ler *Instruções*, comparar a câmera ao vivo com a *Imagem de comparação*, executar até 8 *Ações* e preencher obrigatoriamente o campo *Resolução* (hoje limitado a 16 respostas prontas estáticas).
- **Como conecta:**
  - Consome histórico de ocorrências, *Imagem de comparação* e frames ao vivo via API/RTSP.
  - **Geração automática de Resolução:** sugere o texto completo de encerramento com base nas ações executadas e no áudio/vídeo do atendimento.
  - **Detecção automática de câmera fora de posição/sabotagem:** compara continuamente a imagem ao vivo com a *Imagem de comparação* cadastrada na Zona.
  - **Resumo de passagem de turno:** consolida eventos críticos, falhas de dispositivos e SLA de atendimento por operador/conta.

## 3. Agente de Portaria Autônoma por Voz & WhatsApp (VoIP + Situator Visitas)
- **Problema real documentado:** O Situator integra telefonia **uTech (CTI/VoIP)**, **Situator Visitas**, **Chave Virtual**, **QR Code** e **Scond**, mas a liberação de visitantes/entregadores ainda depende de operador humano na central de portaria remota.
- **Como conecta:**
  - Quando o interfone toca ou chega um visitante, um **Agente de Voz com IA** atende via **SIP/VoIP** (ou WhatsApp/Telegram), identifica o visitante (foto + documento + placa via **LPR**), confirma com o morador via push/WhatsApp e aciona a abertura da porta ou emite **Chave Virtual / QR Code** via API/Webhook do Situator.
  - Em casos ambíguos ou de risco (ex.: senha/biometria de coação), escala imediatamente para o operador humano no Situator com todo o contexto transcrito.

## 4. Visão Computacional Avançada em Nuvem via RTSP (`554`)
- **Problema real documentado:** O D-Guard limita o analítico nativo a resoluções entre `640x480` e `800x600` (12–20 FPS) e oferece funções básicas (cerca, barreira, contagem, velocidade, perambulação).
- **Como conecta:** O D-Guard atua como **Servidor RTSP (porta 554)** e **DASH**, permitindo conectar um motor de IA externo sem trocar câmeras ou DVRs existentes:
  - **Varejo / Indústria:** detecção de EPIs, empilhadeira x pedestre, filas no caixa, mapa de calor, leitura de códigos/etiquetas, comportamento suspeito (furto).
  - **Condomínios / Empresas:** detecção de queda de idosos, fumaça/princípio de incêndio visual, briga/agressão, estacionamento irregular.
  - Cada detecção dispara um evento via **Webhook / HTTP/JSON** direto para o Situator.

## 5. Monitoramento Preditivo de Infraestrutura & Saúde de CFTV (AIOps)
- **Problema real documentado:** O D-Guard possui painel de *Auditoria* (CPU, RAM, banda, espaço em disco, câmeras offline, alerta de *"Taxa de gravação insuficiente"*) e o Situator recebe eventos de *Perda de sinal* e *Perda de conexão*.
- **Como conecta:**
  - Coleta logs criptografados/exportados e métricas de saúde das contas.
  - IA prevê falhas de HD antes da perda de gravação, identifica câmeras com quedas intermitentes recorrentes em horários específicos e abre chamados automáticos de manutenção preventiva com diagnóstico provável.

## 6. Inteligência Veicular & Logística sobre LPR
- **Problema real documentado:** O D-Guard já possui módulo **LPR (DGPLPR)**, listas branca/negra e integração com consulta externa (**Oncar Brasil**), além do Situator vincular veículos a condutores e unidades.
- **Como conecta:**
  - Enriquecimento por IA visual: além da placa, classifica **marca, modelo, cor, tipo de veículo** (moto, caminhão, utilitário) e detecta **placas clonadas/inconsistentes** comparando a aparência visual com os dados cadastrados no Situator.
  - Gestão inteligente de docas, frotas e pátios (tempo de permanência, ociosidade, previsão de chegada).

## 7. Auditoria Inteligente de Acessos & Detecção de Anomalias Comportamentais
- **Problema real documentado:** O Situator registra todo o histórico de acessos (biometria, TAG, QR Code, botoeira *"Desconhecido"*) e ocorrências por zona e horário.
- **Como conecta:**
  - Modelos de detecção de anomalias identificam padrões fora do normal: acesso em horário atípico, uso excessivo de botoeira (*Desconhecido*), compartilhamento de crachá (duas entradas seguidas sem saída — *anti-passback* analítico) ou tentativas repetidas de acesso negado.
  - Gera alertas preventivos no Situator antes que um incidente físico ocorra.
