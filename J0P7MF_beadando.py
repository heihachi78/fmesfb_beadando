# %% [markdown]
# https://github.com/process-intelligence-solutions/pm4py/blob/release/notebooks/1_event_data.ipynb

# %% [markdown]
# # Importok

# %%
import pandas as pd
import pm4py
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py.statistics.traces.generic.log.case_statistics as case_statistics
from pm4py.util import constants
from pm4py.visualization.graphs import visualizer as graphs_visualizer
from pm4py.algo.filtering.log.attributes import attributes_filter
import pm4py.statistics.sojourn_time as stl
from pm4py.visualization.dfg import visualizer as dfg_visualization
import seaborn as sns
import matplotlib.pyplot as plt
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.visualization.footprints import visualizer as fp_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

# %% [markdown]
# # Fájl beolvasása

# %%
pd_log = pd.read_csv("4_PMD_Car_Insurance_Claims_Event_Log_1.0.0.csv", sep=",")

# %%
pd_log.head()

# %% [markdown]
# A Claim ID használható egyedi azonosítóként, az Event az esemény típusa, a Timestamp pedig az esemény időpontja. A User Name a folyamat által használt erőforrás.

# %% [markdown]
# # Átalakítások sorba rendezéshez és xes formátumhoz

# %%
pd_log["claim_id"] = pd_log["Claim ID"].str.replace("CLAIM", "", regex=False).astype(int)
pd_log["activity_key"] = pd_log["Event"]
pd_log["timestamp_key"] = pd.to_datetime(pd_log["Timestamp"], format="%d/%m/%Y %H:%M:%S")
pd_log["org:resource"] = pd_log["User Name"]
pd_log["case_id"] = pd_log["Claim ID"]

# %%
pd_log = pd_log.sort_values(by=["claim_id","timestamp_key"])
pd_log = pd_log.reset_index(drop=True)

# %%
pd_log.head()

# %%
log = pm4py.format_dataframe(pd_log, 
                             case_id="case_id",
                             activity_key="activity_key", 
                             timestamp_key="timestamp_key",
                             timest_format="%d/%m/%Y %H:%M:%S")

# %%
log.head()

# %% [markdown]
# # Az adatok vizsgálta, hogy helyesek voltak-e az átalakítások

# %%
pm4py.get_start_activities(log)

# %%
pm4py.get_end_activities(log)


# %%
pm4py.get_event_attribute_values(log,"org:resource")

# %%
pm4py.get_event_attribute_values(log,"concept:name")

# %%
min_date = log["Timestamp"].min()
max_date = log["Timestamp"].max()
delta = pd.Timedelta(max_date-min_date)
print(min_date, max_date, delta)

# %% [markdown]
# Az adathalmaz időszaka 01/01/2022 to 31/12/2023 így a fenti adatok jónak tűnnek.

# %% [markdown]
# # Exportálás xes formátumban

# %%
xes_exporter.apply(log,'4_PMD_Car_Insurance_Claims_Event_Log_1.0.0.xes')

# %% [markdown]
# # Adatok statisztikái

# %% [markdown]
# ## Event átlagos hosszának vizsgálata

# %%
parameters = {stl.log.get.Parameters.START_TIMESTAMP_KEY:'time:timestamp'}
stl.log.get.apply(log, parameters)

# %% [markdown]
# Mivel csak egy időpont van és nem start és end időppontok ezért 0 mindegyik.

# %% [markdown]
# ## Case duration vizsgálata

# %%
mcd = case_statistics.get_median_case_duration(log, parameters={"timestamp_key": "time:timestamp"})
print(mcd, mcd / 60 / 60 / 24)

# %%
parameters = {constants.PARAMETER_CONSTANT_TIMESTAMP_KEY:"timestamp_key"}
x,y = case_statistics.get_kde_caseduration(log, parameters=parameters)
gviz = graphs_visualizer.apply_plot(x,y,variant=graphs_visualizer.Variants.CASES)
graphs_visualizer.view(gviz)
graphs_visualizer.save(gviz, 'doc/case_duration.png')

# %% [markdown]
# ## Események időbeli eloszlásának vizsgálata

# %%
d_type = "years"
pm4py.view_events_distribution_graph(log, distr_type=d_type, format="png")
pm4py.save_vis_events_distribution_graph(log, 'doc/event_year_dist.png', distr_type=d_type, format="png")

# %%
d_type = "months"
pm4py.view_events_distribution_graph(log, distr_type=d_type, format="png")
pm4py.save_vis_events_distribution_graph(log, 'doc/event_month_dist.png', distr_type=d_type, format="png")

# %%
d_type = "hours"
pm4py.view_events_distribution_graph(log, distr_type=d_type, format="png")
pm4py.save_vis_events_distribution_graph(log, 'doc/event_hour_dist.png', distr_type=d_type, format="png")

# %% [markdown]
# ## Az eseményekben résztvevő resource-ok vizsgálata

# %%
pm4py.view_dotted_chart(log, format='png', attributes=['concept:name','org:resource'])
pm4py.save_vis_dotted_chart(log, 'doc/dot_resource_event.png', format='png', attributes=['concept:name','org:resource'])

# %% [markdown]
# # Directly Folows Graph

# %%
dfg = dfg_discovery.apply(log)
gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY)
dfg_visualization.view(gviz)
dfg_visualization.save(gviz, 'doc/dfg.png')

# %% [markdown]
# ## Heatmap

# %%
fp_log = footprints_discovery.apply(log, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
df = dict(fp_log['dfg'])
activities = list(fp_log['activities'])
df_mtx = pd.DataFrame(columns=activities,index=activities)
df_mtx = df_mtx.fillna(0)

df_keys = list(df.keys())
for key in df_keys:
    i = key[0]
    j = key[1]
    df_mtx.at[i,j] = df[(i,j)]

df_mtx

# %%
%matplotlib inline

# %%
plt.figure(figsize=(12, 10))
sns.heatmap(df_mtx, cmap="coolwarm", robust=True, annot=True, fmt="0000.0f")
plt.title("Heatmap")
plt.xlabel("Activities")
plt.ylabel("Activities")
plt.savefig('doc/heatmap.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# # Alpha Miner

# %% [markdown]
# ## Petri háló

# %%
log.head()

# %%
am_pnet, am_im, am_fm = pm4py.discover_petri_net_alpha(log)
pm4py.view_petri_net(am_pnet, am_im, am_fm, format='png')
pm4py.save_vis_petri_net(am_pnet, am_im, am_fm, 'doc/alpha_miner_petri.png')

# %% [markdown]
# Furcsa, hogy a folyamatnak ugyan van eleje és vége, de olyan, mintha két részre lenne szakadva. Ez nem tűnik megfelelő hálónak, olyan, mintha két külön folyamat lenne, ami egymással nincs kapcsolatban. A valóságban 11002 esetben került sor a Fraud Investigation-re és minden esetben Claim Settlement követte.

# %%
gviz = pn_visualizer.apply(am_pnet, am_im, am_fm, log, variant=pn_visualizer.Variants.FREQUENCY, parameters={fp_visualizer.Variants.SINGLE.value.Parameters.FORMAT:'png'})
pn_visualizer.view(gviz)
pn_visualizer.save(gviz, 'doc/alpha_miner_petri_freq.png')

# %% [markdown]
# ## Footprint

# %%
#fp_net = footprints_discovery.apply(am_pnet, am_im, am_fm, variant=footprints_discovery.Variants.PETRI_REACH_GRAPH)

# %% [markdown]
# Sajnos ezt nem tudtam lefuttatni, több óra után sem ad eredményt. A furcsa Petri háló lehet az oka, de nem tudom biztosan.

# %%
gviz = fp_visualizer.apply(fp_log, parameters={fp_visualizer.Variants.SINGLE.value.Parameters.FORMAT:'png'})
fp_visualizer.view(gviz)
fp_visualizer.save(gviz, 'doc/alpha_miner_footprint.png')
# %%
fp_net = footprints_discovery.apply(dfg, variant=footprints_discovery.Variants.DFG)
gviz = fp_visualizer.apply(fp_net, parameters={fp_visualizer.Variants.SINGLE.value.Parameters.FORMAT:'png'})
fp_visualizer.view(gviz)
fp_visualizer.save(gviz, 'doc/alpha_miner_footprint_dfg.png')
# %% [markdown]
# ## Átmenetek átlagos ideje

# %%
parameters = {pn_visualizer.Variants.PERFORMANCE.value.Parameters.FORMAT:'png'}
gviz = pn_visualizer.apply(am_pnet, am_im, am_fm, parameters=parameters, variant=pn_visualizer.Variants.PERFORMANCE, log=log)
pn_visualizer.view(gviz)
pn_visualizer.save(gviz, 'doc/alpha_miner_performance.png')

# %% [markdown]
# ## Evaluation

# %% [markdown]
# ### Fitness
# Mennyire írja le a modell a logot? Annak mérőszáma, hogy mennyire képes a modell reprodukálni a logot, mennyire fedi le a modell az eseménynaplóban megfigyelt viselkedést.

# %%
fitness_alpha = pm4py.fitness_token_based_replay(log, am_pnet, am_im, am_fm)
fitness_alpha['log_fitness']

# %% [markdown]
# Ez talán az átlagos Alpha Miner eredményeknél alacsonyabbnak tűnik, de nem gondolom kimagaslóan rossznak.

# %% [markdown]
# ### Precision
# A log és a modell által leírt viselkedés eltérése, azt mutatja, hogy a modell mennyire kerüli a naplóban nem megfigyelt viselkedést.

# %%
precision_alpha = pm4py.precision_token_based_replay(log, am_pnet, am_im, am_fm)
precision_alpha

# %% [markdown]
# Ez a precision szerintem nagyon alacsonynak számít Alpha Miner esetén, hiszen pont ez az egyik erőssége. Azt jelenti hogy túl sok olyan működést enged meg a modell ami a logban nem szerepel.

# %% [markdown]
# ### Generalization
# A modell által leírt lehetséges viselkedés ami nem szerepel a logban, pl jövőbeni várható folyamat.

# %%
gen_alpha = generalization_evaluator.apply(log, am_pnet, am_im, am_fm)
gen_alpha

# %% [markdown]
# Ez kimagaslóan jó eredménynek tűnik, de a precision eredményt látva nem feltétlenül hiszek ebben.

# %% [markdown]
# ### Simplicity
# Helyek átalgos kimenő-bejövő kapcsolatainak száma.

# %%
simp_alpha = simplicity_evaluator.apply(am_pnet)
simp_alpha

# %% [markdown]
# Ez egy átlagos Alpha Miner mérőszámnak tűnik.

# %% [markdown]
# # Inductive Miner

# %% [markdown]
# ## Petri háló

# %%
im_net, im_im, im_fm = pm4py.discover_petri_net_inductive(log)
pm4py.view_petri_net(im_net, im_im, im_fm, format='png')
pm4py.save_vis_petri_net(im_net, im_im, im_fm, 'doc/inductive_miner_petri.png')

# %% [markdown]
# Ezen a Petri hálón nekem sokkal inkább tűnik úgy, hogy lefedi a folyamatot és nincs is két részre szakadva. A Petri-hálóban a fekete téglalapok (telített átmenetekkel) általában láthatatlan átmeneteket jelölnek, amelyeket az Inductive Miner algoritmus vezet be a folyamatstruktúra modellezése során. Ezek az átmenetek nem megfigyelhető tevékenységek az eseménynaplóban, hanem a vezérlési folyamat logikáját segítik.

# %% [markdown]
# ## Generált BPMN

# %%
bpmn_graph = pm4py.discover_bpmn_inductive(log, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
gviz =pm4py.visualization.bpmn.visualizer.apply(bpmn_graph)
pm4py.visualization.bpmn.visualizer.view(gviz)
pm4py.visualization.bpmn.visualizer.save(gviz,'doc/inductive_miner_bpmn.png')

# %% [markdown]
# A generált BPMN ellenben jónak tűnik, nem szakad meg sehol a folyamat, szépen végig követhető rajta hogy mi az események sorrendje.

# %% [markdown]
# ## Evaluation

# %% [markdown]
# ### Fitness

# %%
fitness_inductive = pm4py.fitness_token_based_replay(log, im_net, im_im, im_fm)
fitness_inductive['log_fitness']

# %% [markdown]
# A tökéletes fitness erdmény szerintem gyanúsan jó.

# %% [markdown]
# ### Precision

# %%
precision_ind = pm4py.precision_token_based_replay(log, im_net, im_im, im_fm)
precision_ind

# %% [markdown]
# Ez jóval magasabb érték, mint az Alpha Miner esetében.

# %% [markdown]
# ### Generalization

# %%
gen_ind = generalization_evaluator.apply(log, im_net, im_im, im_fm)
gen_ind

# %% [markdown]
# Ez gyakorlatilag az Alpha Minerrel megegyező eredmény.

# %% [markdown]
# ### Simplicity

# %%
simp_ind = simplicity_evaluator.apply(im_net)
simp_ind

# %% [markdown]
# Ez az eredmény megegyzeik gyakorlatilag az Alpha Miner eredményével.

# %% [markdown]
# # Heuristic Miner

# %% [markdown]
# ### Dependency Treshold

# %%
dthl = [0.25, 0.5, 0.75, 0.99]
for dth in dthl:
    hm_net = pm4py.discover_heuristics_net(log, dependency_threshold=dth)
    pm4py.view_heuristics_net(hm_net, format="png")
    pm4py.save_vis_heuristics_net(hm_net, f'doc/heuristic_dth_{dth}.png')

# %% [markdown]
# Ez egy nagyon szép ábra, áttekinthető és helyesek rajta a számok és a folyamat is.

# %% [markdown]
# Függetlenül a paraméter értékétél ugyan azt az ábrát kapjuk.

# %% [markdown]
# ### Minimum Activity Count

# %%
mac_list = [1000, 5000, 10000, 25000]
for mac in mac_list:
    heu_net_mac = heuristics_miner.apply_heu(log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_ACT_COUNT: mac})
    gviz = hn_visualizer.apply(heu_net_mac)
    hn_visualizer.view(gviz)
    hn_visualizer.save(gviz, f'doc/heuristic_mac_{mac}.png')

# %% [markdown]
# ### Minimum Directly Folows

# %%
mdfg_list = [1000, 5000, 10000, 25000]
for mdfg in mdfg_list:
    heu_net_mdfg = heuristics_miner.apply_heu(log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_DFG_OCCURRENCES: mdfg})
    gviz = hn_visualizer.apply(heu_net_mdfg, parameters={'format':'png'})
    hn_visualizer.view(gviz)
    hn_visualizer.save(gviz, f'doc/heuristic_mdfg_{mdfg}.png')

# %% [markdown]
# Ezek is a vártnak megfelelően néznek ki, a mdfg növelésével egyre kevesebb elem jelenik meg a gráfon, ami a folyamatot egyre magasabb szinten az egyre fontosabb eventekre koncentrálva írja le.

# %% [markdown]
# ## Petri háló

# %%
h_net, h_im, h_fm = pm4py.discover_petri_net_heuristics(log, dependency_threshold=0.5)
pm4py.view_petri_net(h_net, h_im, h_fm, format='png')
pm4py.save_vis_petri_net(h_net, h_im, h_fm, 'doc/heuristics_petri_dth_0.5.png')

# %%
h_net, h_im, h_fm = pm4py.discover_petri_net_heuristics(log, dependency_threshold=0.9)
pm4py.view_petri_net(h_net, h_im, h_fm, format='png')
pm4py.save_vis_petri_net(h_net, h_im, h_fm, 'doc/heuristics_petri_dth_0.9.png')

# %% [markdown]
# Nincs eltérés a generált hálóban.

# %% [markdown]
# ## Evaluation

# %% [markdown]
# ### Fitnemss

# %%
fitness_heur = pm4py.fitness_token_based_replay(log, h_net, h_im, h_fm)
fitness_heur['log_fitness']

# %% [markdown]
# Az Alpha Miner-nél jobb, de az Indictive Miner-nél rosszabb eredmény.

# %% [markdown]
# ### Precision

# %%
precision_heur = pm4py.precision_token_based_replay(log, h_net, h_im, h_fm)
precision_heur

# %% [markdown]
# Az eddigi legmagasabb precision.

# %% [markdown]
# ### Generalization

# %%
gen_heur = generalization_evaluator.apply(log, h_net, h_im, h_fm)
gen_heur

# %% [markdown]
# Az eddigi legalacsonyabb generalization.

# %% [markdown]
# ### Simplicity

# %%
simp_heur = simplicity_evaluator.apply(h_net)
simp_heur

# %% [markdown]
# Minimálisan alacsonyabb érték mint korábban a többi módszernél.

# %% [markdown]
# # Evaluation Comparision

# %%
ev_comp = pd.DataFrame(columns=['Alpha','Inductive','Heuristic'], index=['fitness','precision','generalization','simplicity'])

ev_comp.at['fitness','Alpha'] = round(fitness_alpha['log_fitness'],3)
ev_comp.at['fitness','Inductive'] = round(fitness_inductive['log_fitness'],3)
ev_comp.at['fitness','Heuristic'] = round(fitness_heur['log_fitness'],3)

ev_comp.at['precision','Alpha'] = round(precision_alpha,3)
ev_comp.at['precision','Inductive'] = round(precision_ind,3)
ev_comp.at['precision','Heuristic'] = round(precision_heur,3)

ev_comp.at['generalization','Alpha'] = round(gen_alpha,3)
ev_comp.at['generalization','Inductive'] = round(gen_ind,3)
ev_comp.at['generalization','Heuristic'] = round(gen_heur,3)

ev_comp.at['simplicity','Alpha'] = round(simp_alpha,3)
ev_comp.at['simplicity','Inductive'] = round(simp_ind,3)
ev_comp.at['simplicity','Heuristic'] = round(simp_heur,3)


ev_comp = ev_comp[ev_comp.columns].astype(float)
sns.heatmap(ev_comp, cmap='Greens',robust=True,annot=True,fmt='.2f')
plt.savefig('doc/evaluation_comparision.png',format='png',dpi=300,bbox_inches='tight')

# %%
