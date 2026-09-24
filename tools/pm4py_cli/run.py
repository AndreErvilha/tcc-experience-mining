import sys
import os

# Adiciona o diretório 'src' ao PATH do Python para que o pacote jornada_ux seja reconhecido
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from jornada_ux.main import main

if __name__ == "__main__":
    main()
