# AI Agent Showcase：从智能体搭建到多端接入

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com/)
[![Coze](https://img.shields.io/badge/Coze-Agent-purple)](https://www.coze.cn/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

> 在扣子（Coze）搭建智能体 → 封装为统一 Model Provider → 通过 API 接入 CLI 与微信小程序，并记录完整的踩坑与审核流程。

---

## 📌 项目简介

本项目记录并实现了从零搭建一个 AI 智能体，并将其作为 **Model Provider** 接入自有应用的全过程。  
应用形式按难度递增：**CLI 对话 → Web 应用 → 微信小程序对话**（含审核挑战与替代方案）。

核心技术栈：**扣子（Coze） + FastAPI + Python + 微信小程序**。

---

## 🎬 演示

> 建议放置 GIF 或截图，例如：
> - `assets/demo-cli.gif`：终端对话录屏
> - `assets/demo-miniprogram.png`：小程序对话截图
> - `assets/architecture.png`：整体架构图

![CLI Demo](assets/demo-cli.gif)
![Architecture](assets/architecture.png)

---

## 🏗️ 整体架构

```mermaid
graph LR
    A[用户] -->|CLI / 小程序 / Web| B(自有后端 FastAPI)
    B -->|API Key| C[扣子智能体 API]
    C -->|回复| B
    B -->|JSON| A
