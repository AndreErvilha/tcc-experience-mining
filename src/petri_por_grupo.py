"""Rede de Petri (Inductive Miner) e DFG de cada grupo de cada clusterização.
Uso (da raiz do repositório): python src/petri_por_grupo.py [modelo ...]   (ou run_all.py --petri)
Saída: results/jornadas_por_grupo/<modelo>/grupos/{grupo_NN|ruido}/dfg.png e rede_petri.{png,pnml},
      e results/jornadas_por_grupo/<modelo>/petri_parametros.csv.
Grupos com mais de 100 sessões usam filtro de ruído 0,2 no Inductive Miner (registrado em parametros.csv).
Os logs vêm de data/sessions_represented.csv (sem horários: 1 s entre eventos, a ordem é a real). Requer pm4py e Graphviz.
"""
import pickle,os,sys,time,numpy as np,pandas as pd,pm4py,warnings
warnings.filterwarnings('ignore')
O='results/jornadas_por_grupo/'
def G(m,nome):
    d=O+f'{m}/grupos/{nome}/';os.makedirs(d,exist_ok=True);return d
s=pd.read_csv('data/sessions_represented.csv')
base=pd.Timestamp('2025-01-01')
M=['kmeans_tf_idf','kmeans_minilm','kmeans_gemini','hdbscan_tf_idf','hdbscan_minilm','hdbscan_gemini']
ms=sys.argv[1:] or M
for m in ms:
    lab=np.asarray(pickle.load(open(f'clusters/{m}.pkl','rb'))['cluster_labels']);os.makedirs(O+m,exist_ok=True);par=[]
    for g in sorted(set(lab)):
        t0=time.time();sub=s[lab==g];nome='ruido' if g==-1 else f'grupo_{g:02d}'
        rows=[]
        for _,r in sub.iterrows():
            ev=r.text_representation.split()
            rows+= [(r.cd_sessao,e,base+pd.Timedelta(seconds=i)) for i,e in enumerate(ev)]
        log=pm4py.format_dataframe(pd.DataFrame(rows,columns=['c','a','t']),case_id='c',activity_key='a',timestamp_key='t')
        ruido=0.2 if len(sub)>100 else 0.0
        dfg,i,f=pm4py.discover_dfg(log);pm4py.save_vis_dfg(dfg,i,f,G(m,nome)+'dfg.png')
        n,im,fm=pm4py.discover_petri_net_inductive(log,noise_threshold=ruido)
        pm4py.save_vis_petri_net(n,im,fm,G(m,nome)+'rede_petri.png');pm4py.write_pnml(n,im,fm,G(m,nome)+'rede_petri.pnml')
        par.append(dict(grupo=g,sessoes=len(sub),eventos=len(rows),atividades_distintas=log['a'].nunique(),filtro_ruido_inductive=ruido,lugares=len(n.places),transicoes=len(n.transitions)))
        print(m,nome,len(sub),ruido,round(time.time()-t0,1),flush=True)
    pd.DataFrame(par).to_csv(O+f'{m}/petri_parametros.csv',sep=';',index=False,encoding='utf-8-sig')
