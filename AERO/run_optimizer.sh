#!/bin/bash

# 🚀 Запуск веб-интерфейса Aircraft Designer с оптимизацией крыла

# Цвета для вывода
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🛩️  Aircraft Designer с системой оптимизации крыла        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Определить пути проекта независимо от текущей директории
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCAT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCAT_ROOT}/.." && pwd)"

# Найти виртуальное окружение
if [ -d "${WORKSPACE_ROOT}/tf-env" ]; then
    VENV_PATH="${WORKSPACE_ROOT}/tf-env"
elif [ -d "${SCAT_ROOT}/tf-env" ]; then
    VENV_PATH="${SCAT_ROOT}/tf-env"
elif [ -d "${SCRIPT_DIR}/tf-env" ]; then
    VENV_PATH="${SCRIPT_DIR}/tf-env"
else
    echo -e "${YELLOW}⚠️  Виртуальное окружение не найдено${NC}"
    echo "Активируйте tf-env перед запуском:"
    echo "  source ${WORKSPACE_ROOT}/tf-env/bin/activate"
    exit 1
fi

# Активировать виртуальное окружение

# Убедиться, что python видит пакет SCAT при запуске из любой папки
export PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH}"
cd "${WORKSPACE_ROOT}" || exit 1

echo -e "${GREEN}✓ Виртуальное окружение активировано${NC}"

# Проверить требуемые пакеты
echo ""
echo "📦 Проверка зависимостей..."

python3 -c "import SCAT.AERO.main" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Некоторые модули не найдены${NC}"
    echo "Проверьте PYTHONPATH и структуру проекта"
fi

# Предложить выбор режима запуска
echo ""
echo -e "${BLUE}Выберите режим запуска:${NC}"
echo ""
echo "1) 🌐 Веб-интерфейс (основной режим)"
echo "2) 📊 Сгенерировать датасет v2 для ML"
echo "3) 🤖 Обучить ML v2 модель"
echo "4) 🔍 Анализ датасета"
echo "5) 🚀 Быстрый старт (все этапы)"
echo ""
read -p "Выберите (1-5): " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}▶ Запуск веб-интерфейса...${NC}"
        echo "  • Сервер запущен на: http://127.0.0.1:8765"
        echo "  • Остановить: Ctrl+C"
        echo ""
        python3 -m SCAT.AERO.main --port 8765 --no-browser
        ;;
    2)
        echo ""
        echo -e "${GREEN}▶ Генерация датасета...${NC}"
        read -p "Количество примеров (по умолчанию 150): " samples
        samples=${samples:-150}
        python3 -m SCAT.AERO.optimizer sample-v2 \
            --samples $samples \
            --output "${SCRIPT_DIR}/wing_dataset_v2.jsonl" \
            --velocity 50.0 \
            --target-cl 0.55
        ;;
    3)
        echo ""
        echo -e "${GREEN}▶ Обучение ML модели...${NC}"
        
        if [ ! -f "${SCRIPT_DIR}/wing_dataset_v2.jsonl" ]; then
            echo -e "${YELLOW}⚠️  Датасет не найден${NC}"
            echo "Сначала сгенерируйте датасет (опция 2)"
            exit 1
        fi
        
        read -p "Количество эпох (по умолчанию 100): " epochs
        epochs=${epochs:-100}
        
        python3 -m SCAT.AERO.ml_wing_model_v2 train \
            --dataset "${SCRIPT_DIR}/wing_dataset_v2.jsonl" \
            --output "${SCRIPT_DIR}/wing_model_v2.keras" \
            --epochs $epochs
        ;;
    4)
        echo ""
        echo -e "${GREEN}▶ Анализ датасета...${NC}"
        
        if [ ! -f "${SCRIPT_DIR}/wing_dataset_v2.jsonl" ]; then
            echo -e "${YELLOW}⚠️  Датасет не найден${NC}"
            echo "Сначала сгенерируйте датасет (опция 2)"
            exit 1
        fi
        
        read -p "Создавать графики? (y/n, по умолчанию y): " plot
        plot=${plot:-y}
        
        if [ "$plot" = "y" ]; then
            python3 -m SCAT.AERO.analyze_wing \
                --dataset "${SCRIPT_DIR}/wing_dataset_v2.jsonl" \
                --plot \
                --top-k 15
        else
            python3 -m SCAT.AERO.analyze_wing \
                --dataset "${SCRIPT_DIR}/wing_dataset_v2.jsonl" \
                --top-k 15
        fi
        ;;
    5)
        echo ""
        echo -e "${GREEN}▶ Быстрый старт (полная цепочка)...${NC}"
        echo ""
        read -p "Количество примеров (по умолчанию 100): " samples
        samples=${samples:-100}
        read -p "Количество эпох (по умолчанию 80): " epochs
        epochs=${epochs:-80}
        
        python3 "${SCRIPT_DIR}/wing_quickstart.py" --samples $samples --epochs $epochs
        ;;
    *)
        echo -e "${YELLOW}⚠️  Неправильный выбор${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
