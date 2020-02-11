from odl.ufunc_ops.ufunc_ops import log_op
from dival.evaluation import TaskTable
from dival.measure import PSNR, SSIM
from dival.reconstructors.odl_reconstructors import FBPReconstructor
from dival.datasets.standard import get_standard_dataset
from dival.util.constants import MU_MAX
from pathlib import Path

# %% Apenas modificar essas variáveis 
#step (train, test or validation)
step = 'test'
initial_slice = 10 
final_slice = 20


dataset = get_standard_dataset('lodopab', observation_model='pre-log')
ray_trafo = dataset.get_ray_trafo(impl='astra_cpu')
reco_space = ray_trafo.domain

"""
get_data_pairs:
    @args:
        'tipo da base(test,train ou validation)'
        'slice inicial',
        'slice final',
    Os slices definem o intervalo que deverá ser utilizado da base de dados
    Caso não defina um intervalo, toda a base será utilizada
"""
test_data = dataset.get_data_pairs(step, initial_slice, final_slice)

# %% task table and reconstructors
eval_tt = TaskTable()

"""
A configuração filter_type com Ram-Lak e frequency_scaling 1.0 foram as utilizadas no artigo
"""
fbp_reconstructor = FBPReconstructor(
    ray_trafo, hyper_params={
        'filter_type': 'Ram-Lak',
        'frequency_scaling': 1.0},
    pre_processor=(-1/MU_MAX) * log_op(ray_trafo.range))

reconstructors = [fbp_reconstructor]

eval_tt.append_all_combinations(reconstructors=reconstructors,
                                test_data=[test_data])

# %% run task table
results = eval_tt.run()
results.apply_measures([PSNR, SSIM])
print(results)

# %% plot reconstructions


Path(step).mkdir(parents=True, exist_ok=True)
Path(step + '_gt').mkdir(parents=True, exist_ok=True)


"""Plot all reconstructions.

        Parameters
        ----------
        step :          (train, test or validation)
        test_ind :      range(x); quantidade de plots que serão salvos
                        se test_ind for igual a -1, serão plotados todos os elementos
        initial_slice:  parte inicial das amostras  
"""
fig = results.plot_all_reconstructions(step, initial_slice, test_ind=-1,
                                       fig_size=(9, 4), vrange='individual')