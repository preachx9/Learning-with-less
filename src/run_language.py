"""Use the unchanged Yorùbá experiment with a separate dataset per language."""
from pathlib import Path
import argparse
import data_utils
import study_engine

LANGUAGES={'yor':'Yorùbá','ibo':'Igbo','hau':'Hausa','pcm':'Nigerian Pidgin'}

def run_language(code):
    if code not in LANGUAGES:raise ValueError(code)
    root=Path(__file__).resolve().parent/'languages'/code
    root.mkdir(parents=True,exist_ok=True)
    data_utils.ROOT=root
    data_utils.BASE=f'https://huggingface.co/datasets/masakhane/masakhanews/resolve/{data_utils.REVISION}/data/{code}/'
    study_engine.ROOT=root
    study_engine.OUT=root/'results'
    study_engine.OUT.mkdir(exist_ok=True)
    study_engine.PROTOCOL.update({'language_code':code,'language':LANGUAGES[code],
        'study_history':'Extension selected after observing the Yorùbá result. Model settings, sampling and overlap checks unchanged. Separate monolingual models; no pooled multilingual model.'})
    print('LANGUAGE',LANGUAGES[code],flush=True)
    study_engine.main()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('language',choices=LANGUAGES)
    run_language(parser.parse_args().language)
