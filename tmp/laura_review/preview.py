from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sqlite3
from contextlib import contextmanager
import streamlit as st
from ui.laura_plan import render_laura_plan
st.set_page_config(page_title='Laura Gao Plan preview', layout='wide')
@contextmanager
def connection():
    with sqlite3.connect(Path(__file__).with_name('preview.db')) as conn:
        yield conn
render_laura_plan({'username': 'preview'}, {}, connection)
