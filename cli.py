# cli.py — the `sofia "..."` entry point
import sys
from graph import graph

def main():
    task = " ".join(sys.argv[1:])
    if not task:
        print('usage: sofia "your task"')
        return
    print(graph.invoke({"task": task})["result"])

if __name__ == "__main__":
    main()
