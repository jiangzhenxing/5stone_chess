from chess_board import  ChessBoard
import logging.config

logging.config.fileConfig('logging.conf')

ChessBoard().launch()

