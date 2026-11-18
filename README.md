# Seventh AI — Oportunidades de Produtos com IA para D-Guard & Situator

Este projeto mapeia oportunidades práticas e arquiteturas de produtos baseados em Inteligência Artificial conectados ao ecossistema **Seventh** (**D-Guard VMS** e **Situator PSIM**), com base nos manuais oficiais:

- **D-Guard** (Versão 7.1.4.9 | Web 1.46.0)
- **Situator** (Versão 5)

## Pontos de Integração Nativos Encontrados nos Manuais

### D-Guard (VMS)
- **Servidor RTSP (porta padrão `554`)** e **DASH**: permite consumir streams de vídeo ao vivo em pipelines externos de Visão Computacional.
- **API REST (porta padrão `8081`)**: administração e integração com sistemas terceiros.
- **Ação `Enviar requisição HTTP GET`** (módulo *Eventos*): dispara webhooks/chamadas para backends de IA quando ocorre detecção de movimento, LPR, cerca virtual, etc.
- **Exportação periódica de Snapshots via FTP**: envio automático de frames em intervalo configurável para análise assíncrona.
- **LPR + Integração externa (ex.: Oncar Brasil)** e **Reconhecimento Fácil**.
- **D-Guard Intelligence**: motor de regras que aceita variáveis de entrada/saída e notifica o Situator.

### Situator (PSIM)
- **API de Recepção de Eventos (HTTP/JSON)** e perfil **Cliente da API**: permite que motores de IA injetem ocorrências já classificadas com prioridade, conta e zona.
- **Webhooks** (`Configurações > Integrações > Webhooks`): recepção de eventos de plataformas terceiras via URL + Token.
- **Fluxo de Atendimento**: filas (*Aberto*, *Atendimento*, *Espera*), *Instruções*, *Ações rápidas*, *Resoluções* (até 16 respostas prontas) e *Imagem de comparação*.
- **Telefonia VoIP / CTI (uTech)**, **E-mail (SMTP)**, **SMS (Zenvia)**, **Scond**, **Situator Visitas** e **Chave Virtual**.

## Principais Oportunidades de Produtos com IA

Veja a análise completa em [docs/oportunidades-ia.md](docs/oportunidades-ia.md) ou acesse a interface web publicada na Vercel.
