# AI Agent Showcase：从智能体搭建到多端接入

[![Coze](https://img.shields.io/badge/Coze-Agent-purple)](https://www.coze.cn/)
[![OneAPI](https://img.shields.io/badge/OneAPI-Gateway-orange)](https://github.com/songquanpeng/one-api)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

> 在扣子（Coze）搭建智能体 → 通过 OneAPI 包装为统一 Model Provider → 以 OpenAI 兼容接口接入 CLI 与微信小程序。

---

## 📌 项目简介

本项目记录并实现了从零搭建 AI 智能体，并将其作为 **Model Provider** 接入自有应用的全过程。

**核心链路**：扣子智能体 → OneAPI 统一网关 → CLI / 小程序对话。

---

## 🎬 演示

![CLI Demo](assets/demo-cli.gif)
![Architecture](assets/architecture.png)

---

## 🏗️ 整体架构

```mermaid
graph LR
    A[CLI] --> D[OneAPI 网关]
    B[微信小程序] --> D
    C[Web / 其他] --> D
    D -->|OpenAI 兼容接口| E[扣子智能体 API]
    E -->|回复| D
    D -->|JSON| A
