PT = {'radius':'raio','texture':'textura','perimeter':'perímetro','area':'área','smoothness':'suavidade','compactness':'compacidade',
      'concavity':'concavidade','concave points':'pontos côncavos','symmetry':'simetria','fractal dimension':'dimensão fractal'}
def nome_atributo(f):
    g = 'pior ' if f.startswith('worst') else 'erro padrão de ' if f.endswith('error') else 'média de '
    b = f.replace('mean ', '').replace('worst ', '').replace(' error', ''); return g + PT.get(b, b)
