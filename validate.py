import json, jsonschema, sys, os
schema = json.load(open('D:/wmed/emsim/transitions/schema.json'))
d = 'D:/wmed/emsim/transitions/Endocrine/'
for f in sorted(os.listdir(d)):
    if f.endswith('.json'):
        try:
            data = json.load(open(d+f, encoding='utf-8'))
            jsonschema.validate(data, schema)
            print('OK', len(data['pairs']), f)
        except Exception as e:
            print('FAIL', f, type(e).__name__, str(e)[:200])
