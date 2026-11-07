# Resource-Efficient Latent Diffusion Framework for Medical Image Synthesis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Магистерская диссертация по образовательной программе **7M06108 «Прикладной искусственный интеллект» (Applied Artificial Intelligence)**, Astana IT University (2026–2028).

* **Магистрант:** Кубесов Ерлан Русланович (Kubesov Erlan, группа AAI-2605M)
* **Научный руководитель:** Ryspayeva Marya, PhD, Assistant Professor
* **Тема:** Разработка ресурсоэффективного фреймворка на основе латентной диффузии для синтеза медицинских изображений
* **Title (EN):** Development of a Resource-Efficient Latent Diffusion Framework for Medical Image Synthesis
* **Тақырып (KZ):** Медициналық кескіндерді синтездеуге арналған ресурстық тиімді латентті диффузиялық фреймворкты әзірлеу

---

## 1. Идея и исследовательские вопросы (Research Questions)

Основная идея исследования — генерировать синтетические 2D аксиальные КТ-срезы органов брюшной полости, моделируя диффузионный процесс не в пиксельном пространстве, а в компактном латентном представлении автоэнкодера (VAE), с оптимизацией распределения шагов обучения.

* **RQ1:** Насколько латентная диффузия (LDM) дешевле прямого пиксельного DDPM при сопоставимом визуальном качестве и идентичном аппаратном обеспечении?
* **RQ2:** Какой компромисс между качеством реконструкции VAE, качеством диффузионного синтеза и вычислительной сложностью обеспечивает коэффициент сжатия ($\times4$ vs $\times8$)?
* **RQ3:** Позволяет ли адаптивное распределение обучающих шагов по уровням шума (**CBATS — Compute-Budget-Aware Timestep Sampling**) улучшить качество генерации при строго фиксированном вычислительном бюджете (GPU-hours)?

---

## 2. Данные (Dataset)

Используется стандартизованный биомедицинский бенчмарк **OrganAMNIST** из коллекции **MedMNIST+** ([Zenodo 10519652](https://zenodo.org/records/10519652)):
* **Модальность:** 2D аксиальные срезы компьютерной томографии (КТ), grayscale.
* **Классы:** 11 анатомических областей (мочевой пузырь, бедренные кости, сердце, почки, печень, легкие, поджелудочная железа, селезенка).
* **Объем выборки:** 34 561 (*train*) / 6 491 (*val*) / 17 778 (*test*), суммарно 58 830 изображений.
* **Разрешение:** Пилотный запуск — $64 \times 64$, основной целевой эксперимент — $128 \times 128$.

---

## 3. Матрица контролируемых экспериментов (E0–E4)

| Код | Модель / Режим | Описание | Роль в исследовании |
| :--- | :--- | :--- | :--- |
| **E0** | Pixel DDPM | Прямая диффузия в пикселях ($64\times64$) | Контроль вычислительных затрат |
| **E1** | Standard LDM | LDM с равномерным (*uniform*) выбором timesteps $t \sim \mathcal{U}(1, T)$ | Базовый LDM baseline |
| **E2** | LDM + Fixed Sampling | LDM с фиксированным нелинейным расписанием timesteps | Контроль фактора неравномерности |
| **E3** | **Budget-aware LDM (CBATS)** | Адаптивная выборка сложных timesteps по скользящей EMA loss с поправкой важности | **Основная предлагаемая модификация** |
| **E4** | Latent Compression Ablation | Сравнение латентного фактора сжатия $\times4$ vs $\times8$ | Исследование компромисса VAE |

---

## 4. Структура репозитория

```text
├── configs/                     # YAML конфигурации гиперпараметров (E0–E4)
│   ├── vae_f4.yaml
│   ├── pixel_ddpm_e0.yaml
│   ├── ldm_uniform_e1.yaml
│   └── ldm_cbats_e3.yaml
├── data/raw/                    # Локальные .npz файлы OrganAMNIST (в .gitignore)
├── checkpoints/                 # Сохраненные веса моделей .pt (в .gitignore)
├── docs/                        # Официальные документы (ИПРМ, план диссертации)
├── src/                         # Исходный модульный код фреймворка
│   ├── data/                    # Dataset, DataLoader, нормализация [-1, 1]
│   ├── models/                  # VAE, pixel U-Net, latent U-Net
│   ├── schedulers/              # DDPM, DDIM инференс, алгоритм CBATS
│   ├── metrics/                 # MAE, SSIM, FID, KID, VRAM профилирование
│   └── utils/                   # Фиксация seed, логирование, сетки изображений
├── scripts/                     # Точки входа (CLI) для обучения и оценки
│   └── audit_organamnist.py     # Загрузка и первичный статистический аудит
├── notebooks/                   # Jupyter ноутбуки для EDA и проверки гипотез
│   └── 01_organamnist_audit.ipynb
├── experiments/samples/         # Артефакты генерации и кривые Парето
├── requirements.txt
└── README.md
```

---

## 5. Быстрый старт (Quickstart)

```bash
# 1. Клонирование и переход в репозиторий
git clone git@github.com:Erlan1502/resource-efficient-medical-ldm.git
cd resource-efficient-medical-ldm

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Загрузка данных и первичный аудит (EDA)
python scripts/audit_organamnist.py --size 64
```

---

## 6. Календарный план публикаций и защиты

* **Февраль 2027 г.:** Тезис / доклад *Preliminary Study of Resource-Efficient Latent Diffusion Models for Medical Image Synthesis* на 2027 IEEE 4th International Student Conference on Digital Generation (AITU).
* **Май 2027 г.:** Доклад *Resource-Efficient Latent Diffusion Framework for Medical Image Synthesis* на конференции IEEE SIST 2027.
* **Сентябрь 2027 г.:** Статья в рецензируемом журнале *Scientific Journal of Astana IT University* (КОКСНВО МНВО РК).
* **Июнь 2028 г.:** Защита магистерской диссертации.
