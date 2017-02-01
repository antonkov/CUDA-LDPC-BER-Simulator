import pandas as pd

ferReport = pd.read_csv('report50iter', delim_whitespace=True, header=1)
spectrumReport = pd.read_csv('spectrumReport', delim_whitespace=True, header=1)
df = pd.merge(ferReport, spectrumReport, on=['Filename','Filename'])
df = df.sort_values(by=['FER%'])
spectrums = ['spectrum1', 'spectrum2', 'spectrum3']
df = df[['FER%'] + spectrums]
grouped = df.groupby(spectrums)
with open("asPlot", "w") as out:
    for name, group in grouped:
        out.write(str(name) + '\n')
        v = group['FER%']
        out.write(str(len(v)) + '\n')
        for val in v:
            out.write(str(val) + '\n')
df.to_csv('resultTable',
        sep=' ',
        index=None)
