import json

maj_path = '/home/datht/mats/sql_writer/evaluation/majority_voting_results.json'
spi_eval_path = '/home/datht/mats/sql_writer/evaluation/evaluation_spider_dev/GRPO-3B-cp2000-n50/evaluation.json'

def norm(s):
    if s is None:
        return ''
    s = str(s).replace('\r',' ').replace('\n',' ')
    return ' '.join(s.split()).strip()

with open(maj_path, 'r', encoding='utf-8') as f:
    maj = json.load(f)
maj_results = maj.get('results', [])

with open(spi_eval_path, 'r', encoding='utf-8') as f:
    spi_list = json.load(f)
spi_map = {}
for e in spi_list:
    db = str(e.get('db_id',''))
    pred = norm(e.get('pred',''))
    ex = 1 if e.get('exec_result') else 0
    key = (db, pred)
    if key not in spi_map or ex == 0:
        spi_map[key] = ex

mismatch = []
for r in maj_results:
    db = str(r.get('db_id',''))
    sel = norm(r.get('selected_sql',''))
    if not bool(r.get('is_sample_correct', False)):
        continue
    ex = spi_map.get((db, sel), None)
    if ex == 0:
        mismatch.append({
            'sample_id': r.get('sample_id'),
            'db_id': db,
            'question': (r.get('question','') or '')[:200],
            'selected_sql': sel[:500]
        })

print('Total majority-correct but spider-incorrect:', len(mismatch))
for i, m in enumerate(mismatch[:100], 1):
    print('[{}] sample_id={} db={}'.format(i, m['sample_id'], m['db_id']))
    print('  SQL:', m['selected_sql'])
    print('  Q:', m['question'])
if len(mismatch) > 100:
    print('... and {} more'.format(len(mismatch)-100))
