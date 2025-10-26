from env.env import play
from UCS_Al.UCS import ucs
from BFS_Al.BFS import bfs
from DLS_Al.DLS import dls
from Astar_Al.Astar import astar


if __name__ == "__main__":
    play('medium', astar, delay=500)