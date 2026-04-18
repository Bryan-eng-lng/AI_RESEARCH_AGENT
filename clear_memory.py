"""
Utility to clear research memory and checkpoints.
Usage:
  python clear_memory.py          # clears everything
  python clear_memory.py --checkpoints  # clears only LangGraph checkpoints
  python clear_memory.py --rag          # clears only RAG vector memory
"""
import sys
import sqlite3
import shutil
import os

def clear_checkpoints():
    """Clear LangGraph checkpoint state from research_memory.db"""
    conn = sqlite3.connect("research_memory.db")
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in c.fetchall()]
    if not tables:
        print("  Checkpoints: already empty")
    for table in tables:
        c.execute(f"DELETE FROM [{table}]")
        print(f"  Cleared table: {table}")
    conn.commit()
    conn.close()
    print("  Checkpoints cleared.")

def clear_rag():
    """Clear ChromaDB vector store"""
    vector_path = "./vector_store"
    if os.path.exists(vector_path):
        shutil.rmtree(vector_path)
        os.makedirs(vector_path)
        print("  RAG vector store cleared.")
    else:
        print("  RAG vector store: not found, nothing to clear.")

def show_status():
    """Show current memory status"""
    print("\n=== MEMORY STATUS ===")
    # Checkpoints
    conn = sqlite3.connect("research_memory.db")
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in c.fetchall()]
    print(f"Checkpoint tables: {tables if tables else 'empty'}")
    for table in tables:
        c.execute(f"SELECT COUNT(*) FROM [{table}]")
        print(f"  {table}: {c.fetchone()[0]} rows")
    conn.close()

    # RAG
    vector_path = "./vector_store"
    if os.path.exists(vector_path):
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(vector_path)
            for f in files
        )
        print(f"RAG vector store: {size / 1024:.1f} KB")
    else:
        print("RAG vector store: not found")
    print("=====================\n")

if __name__ == "__main__":
    args = sys.argv[1:]
    show_status()

    if not args or "--all" in args:
        print("Clearing everything...")
        clear_checkpoints()
        clear_rag()
    elif "--checkpoints" in args:
        print("Clearing checkpoints only...")
        clear_checkpoints()
    elif "--rag" in args:
        print("Clearing RAG memory only...")
        clear_rag()

    print("\nDone. Run status after clear:")
    show_status()
