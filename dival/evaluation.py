# -*- coding: utf-8 -*-
"""Tools for the evaluation of reconstruction methods.
"""
import sys
from warnings import warn
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from odl.solvers.util.callback import Callback, CallbackStore
from dival.util.plot import plot_image, plot_images
from dival.util.std_out_err_redirect_tqdm import std_out_err_redirect_tqdm
from dival.measure import Measure
from dival.data import DataPairs
from dival import IterativeReconstructor, LearnedReconstructor


class TaskTable:
    """Task table containing reconstruction tasks to evaluate.

    Attributes
    ----------
    name : str
        Name of the task table.
    tasks : list of dict
        Tasks that shall be run. The fields of each dict are set from the
        parameters to :meth:`append` (or :meth:`append_all_combinations`). Cf.
        documentation of :meth:`append` for details.
    results : :class:`ResultTable` or `None`
        Results from the latest call to :meth:`run`.
    """
    def __init__(self, name=''):
        self.name = name
        self.tasks = []
        self.results = None

    def run(self, save_reconstructions=True, show_progress='text'):
        """Run all tasks and return the results.

        The returned :class:`ResultTable` object is also stored as
        :attr:`results`.

        Parameters
        ----------
        save_reconstructions : bool, optional
            Whether the reconstructions should be saved in the results.
            The default is ``True``.

            If measures shall be applied after this method returns, it must be
            ``True``.

            If ``False``, no iterates (intermediate reconstructions) will be
            saved, even if ``task['options']['save_iterates']==True``.
        show_progress : str, optional
            Whether and how to show progress. Options are:

                ``'text'`` (default)
                    print a line before running each task
                ``'tqdm'``
                    show a progress bar with ``tqdm``
                `None`
                    do not show progress

        Returns
        -------
        results : :class:`ResultTable`
            The results.
        """
        row_list = []
        with std_out_err_redirect_tqdm(None if show_progress == 'tqdm' else
                                       sys.stdout) as orig_stdout:
            for i, task in enumerate(tqdm(self.tasks,
                                          desc='task',
                                          file=orig_stdout,
                                          disable=(show_progress != 'tqdm'))):
                if show_progress == 'text':
                    print('running task {i}/{num_tasks} ...'.format(
                        i=i, num_tasks=len(self.tasks)))
                test_data = task['test_data']
                reconstructor = task['reconstructor']
                if test_data.ground_truth is None and task['measures']:
                    raise ValueError('missing ground truth, cannot apply '
                                     'measures')
                measures = [(measure if isinstance(measure, Measure) else
                             Measure.get_by_short_name(measure))
                            for measure in task['measures']]
                options = task['options']
                skip_training = options.get('skip_training', False)
                save_iterates = (save_reconstructions and
                                 options.get('save_iterates'))

                hp_choices = task.get('hyper_param_choices')
                if hp_choices:
                    # run all hyper param choices as sub-tasks
                    retrain_param_keys = [k for k, v in
                                          reconstructor.HYPER_PARAMS.items()
                                          if v.get('retrain', False)]
                    orig_hyper_params = reconstructor.hyper_params.copy()

                    def _warn_if_invalid_keys(keys):
                        for k in keys:
                            if k not in reconstructor.HYPER_PARAMS.keys():
                                warn("choice for unknown hyper parameter '{}' "
                                     "for reconstructor of type '{}' will be "
                                     'ignored'.format(k, type(reconstructor)))

                    if isinstance(hp_choices, dict):
                        _warn_if_invalid_keys(hp_choices.keys())
                        keys_retrain_first = sorted(
                            hp_choices.keys(),
                            key=lambda k: k not in retrain_param_keys)
                        param_values = [
                            hp_choices.get(k, [orig_hyper_params[k]]) for k in
                            keys_retrain_first]
                        hp_choice_list = [
                            dict(zip(keys_retrain_first, v)) for
                            v in product(*param_values)]
                    else:
                        hp_choice_list = hp_choices
                        for hp_choice in hp_choice_list:
                            _warn_if_invalid_keys(hp_choice.keys())
                    for j, hp_choice in enumerate(
                            tqdm(hp_choice_list, desc='sub-task',
                                 file=orig_stdout,
                                 disable=(show_progress != 'tqdm'),
                                 leave=False)):
                        if show_progress == 'text':
                            print('sub-task {j}/{n} ...'
                                  .format(j=j, n=len(hp_choice_list)))
                        train = (isinstance(reconstructor,
                                            LearnedReconstructor) and (
                            j == 0 or any(
                                (hp_choice.get(k, orig_hyper_params[k]) !=
                                 reconstructor.hyper_params[k]
                                 for k in retrain_param_keys))))
                        reconstructor.hyper_params = orig_hyper_params.copy()
                        reconstructor.hyper_params.update(hp_choice)
                        if train and not skip_training:
                            reconstructor.train(task['dataset'])
                        row = self._run_task(
                            reconstructor=reconstructor,
                            test_data=test_data,
                            measures=measures,
                            hp_choice=hp_choice,
                            options=options,
                            save_reconstructions=save_reconstructions,
                            save_iterates=save_iterates,
                            )
                        row['task_ind'] = i
                        row['sub_task_ind'] = j
                        row_list.append(row)
                    reconstructor.hyper_params = orig_hyper_params.copy()
                else:
                    # run task (with hyper params as they are)
                    if (isinstance(reconstructor, LearnedReconstructor) and
                            not skip_training):
                        reconstructor.train(task['dataset'])

                    row = self._run_task(
                        reconstructor=reconstructor,
                        test_data=test_data,
                        measures=measures,
                        hp_choice=None,
                        options=options,
                        save_reconstructions=save_reconstructions,
                        save_iterates=save_iterates,
                        )
                    row['task_ind'] = i
                    row['sub_task_ind'] = 0
                    row_list.append(row)

        self.results = ResultTable(row_list)
        return self.results

    def _run_task(self, reconstructor, test_data, measures, options, hp_choice,
                  save_reconstructions, save_iterates):
        reconstructions = []
        if (isinstance(reconstructor, IterativeReconstructor) and
                save_iterates):
            iterates = []
            if options.get('save_iterates_measure_values'):
                iterates_measure_values = {m.short_name: []
                                           for m in measures}

        for observation, ground_truth in zip(test_data.observations,
                                             test_data.ground_truth):
            if isinstance(reconstructor, IterativeReconstructor):
                callbacks = []
                orig_callback = reconstructor.callback
                if isinstance(orig_callback, Callback):
                    callbacks.append(orig_callback)
                elif orig_callback is not None:
                    warn('original callback of reconstructor '
                         'deactivated since it is not of type '
                         '`odl.solvers.util.callback.Callback` (got '
                         '`{}`)'.format(type(orig_callback)))
                if save_iterates:
                    iters = []
                    iterates.append(iters)
                    callback_save_iterates = CallbackStore(
                        iters,
                        step=options.get('save_iterates_step', 1))
                    callbacks.append(callback_save_iterates)
                if options.get('save_iterates_measure_values'):
                    for measure in measures:
                        iters_mvs = []
                        iterates_measure_values[
                            measure.short_name].append(iters_mvs)
                        callback_store = CallbackStore(
                            iters_mvs,
                            step=options.get('save_iterates_step', 1))
                        callbacks.append(
                            callback_store *
                            measure.as_operator_for_fixed_ground_truth(
                                ground_truth))
                if len(callbacks) > 0:
                    reconstructor.callback = callbacks[-1]
                    for callback in callbacks[-2::-1]:
                        reconstructor.callback &= callback

            reconstructions.append(reconstructor.reconstruct(
                observation))

            if isinstance(reconstructor, IterativeReconstructor):
                reconstructor.callback = orig_callback

        measure_values = {}
        for measure in measures:
            measure_values[measure.short_name] = [
                measure.apply(r, g) for r, g in zip(
                    reconstructions, test_data.ground_truth)]
        misc = {}
        if isinstance(reconstructor, IterativeReconstructor):
            if save_iterates:
                misc['iterates'] = iterates
            if options.get('save_iterates_measure_values'):
                misc['iterates_measure_values'] = (
                    iterates_measure_values)
        if hp_choice:
            misc['hp_choice'] = hp_choice

        row = {'reconstructions': reconstructions,
               'reconstructor': reconstructor,
               'test_data': test_data,
               'measure_values': measure_values,
               'misc': misc}
        if save_reconstructions:
            row['reconstructions'] = reconstructions
        return row

    def append(self, reconstructor, test_data, measures=None, dataset=None,
               hyper_param_choices=None, options=None):
        """Append a task.

        Parameters
        ----------
        reconstructor : :class:`.Reconstructor`
            The reconstructor.
        test_data : :class:`.DataPairs`
            The test data.
        measures : sequence of (:class:`.Measure` or str)
            Measures that will be applied. Either :class:`.Measure` objects or
            their short names can be passed.
        dataset : :class:`.Dataset`
            The dataset that will be passed to
            :meth:`reconstructor.train <LearnedReconstructor.train>` if it is a
            :class:`.LearnedReconstructor`.
        hyper_param_choices : dict of list or list of dict, optional
            Choices of hyper parameter combinations to try as sub-tasks.

                * If a dict of lists is specified, all combinations of the
                  list elements (cartesian product space) are tried.
                * If a list of dicts is specified, each dict is taken as a
                  parameter combination to try.

            The current parameter values are read from
            :attr:`Reconstructor.hyper_params` in the beginning and used as
            default values for all parameters not specified in the passed
            dicts. Afterwards, the original values are restored.
        options : dict
            Options that will be used. Options are:

            ``'skip_training'`` : bool, optional
                Whether to skip training. Can be used for manual training
                of reconstructors (or loading of a stored state).
                The default is ``False``.
            ``'save_iterates'`` : bool, optional
                Whether to save the intermediate reconstructions of iterative
                reconstructors (the default is ``False``).
                Will be ignored if ``save_reconstructions=False`` is passed to
                `run`. Requires the reconstructor to call its `callback`
                attribute after each iteration.
            ``'save_iterates_measure_values'`` : bool, optional
                Whether to compute and save the measure values for each
                intermediate reconstruction of iterative reconstructors
                (the default is ``False``). Requires the reconstructor to call
                its `callback` attribute after each iteration.
            ``'save_iterates_step'`` : int, optional
                Step size for ``'save_iterates'`` and
                ``'save_iterates_measure_values'`` (the default is 1).
        """
        if measures is None:
            measures = []
        if options is None:
            options = {}
        if (isinstance(reconstructor, LearnedReconstructor) and
                not options.get('skip_training', False) and dataset is None):
            raise ValueError('in order to use a learned reconstructor you '
                             'must specify a `dataset` for training (or set '
                             '``skip_training: True`` in `options` and train '
                             'manually)')
        self.tasks.append({'reconstructor': reconstructor,
                           'test_data': test_data,
                           'measures': measures,
                           'dataset': dataset,
                           'hyper_param_choices': hyper_param_choices,
                           'options': options})

    def append_all_combinations(self, reconstructors, test_data, measures=None,
                                datasets=None, hyper_param_choices=None,
                                options=None):
        """Append tasks of all combinations of test data, reconstructors and
        optionally datasets.
        The order is taken from the lists, with test data changing slowest
        and reconstructor changing fastest.

        Parameters
        ----------
        reconstructors : list of `Reconstructor`
            Reconstructor list.
        test_data : list of `DataPairs`
            Test data list.
        measures : sequence of (`Measure` or str)
            Measures that will be applied. The same measures are used for all
            combinations of test data and reconstructors. Either `Measure`
            objects or their short names can be passed.
        datasets : list of `Dataset`, optional
            Dataset list. Required if `reconstructors` contains at least one
            `LearnedReconstructor`.
        hyper_param_choices : list of (dict of list or list of dict), optional
            Choices of hyper parameter combinations for each reconstructor,
            which are tried as sub-tasks.
            The i-th element of this list is used for the i-th reconstructor.
            See `append` for documentation of how the choices are passed.
        options : dict
            Options that will be used. The same options are used for all
            combinations of test data and reconstructors. See `append` for
            documentation of the options.
        """
        if datasets is None:
            datasets = [None]
        if hyper_param_choices is None:
            hyper_param_choices = [None] * len(reconstructors)
        for test_data_ in test_data:
            for dataset in datasets:
                for reconstructor, hp_choices in zip(reconstructors,
                                                     hyper_param_choices):
                    self.append(reconstructor=reconstructor,
                                test_data=test_data_,
                                measures=measures,
                                dataset=dataset,
                                hyper_param_choices=hp_choices,
                                options=options)

    def __repr__(self):
        return "TaskTable(name='{name}', tasks={tasks})".format(
            name=self.name,
            tasks=self.tasks)


class ResultTable:
    """The results of a :class:`.TaskTable`.

    Cf. :attr:`TaskTable.results`.

    Attributes
    ----------
    results : :class:`pandas.DataFrame`
        The results.
        The index is given by ``'task_ind'`` and ``'sub_task_ind'``, and the
        columns are ``'reconstructions'``, ``'reconstructor'``,
        ``'test_data'``, ``'measure_values'`` and ``'misc'``.
    """
    def __init__(self, row_list):
        """
        Usually, objects of this type are constructed by
        :meth:`TaskTable.run`, which sets :attr:`TaskTable.results`, rather
        than by manually calling this constructor.

        Parameters
        ----------
        row_list : list of dict
            Result rows.
            Used to build :attr:`results` of type :class:`pandas.DataFrame`.
        """
        self.results = pd.DataFrame(row_list).set_index(['task_ind',
                                                         'sub_task_ind'])

    def apply_measures(self, measures, task_ind=None):
        """Apply (additional) measures to reconstructions.

        This is not possible if the reconstructions were not saved, in which
        case a :class:`ValueError` is raised.

        Parameters
        ----------
        measures : list of :class:`.Measure`
            Measures to apply.
        task_ind : int or sequence of ints, optional
            Indexes of tasks to which the measures shall be applied.
            If `None`, this is interpreted as "all results".

        Raises
        ------
        ValueError
            If reconstructions are missing or `task_ind` is not valid.
        """
        if task_ind is None:
            indexes = self.results.index.levels[0]
        elif np.isscalar(task_ind):
            indexes = [task_ind]
        elif isinstance(task_ind, list):
            indexes = task_ind
        else:
            raise ValueError('`task_ind` must be a scalar, a list of ints or '
                             '`None`')
        for i in indexes:
            rows = self.results.loc[i]
            for j in range(len(rows)):
                row = rows.loc[j]
                if row['reconstructions'] is None:
                    raise ValueError('reconstructions missing in task {}{}'
                                     .format(i, '.{}'.format(j) if
                                                len(rows) > 1 else ''))
                for measure in measures:
                    if isinstance(measure, str):
                        measure = Measure.get_by_short_name(measure)
                    row['measure_values'][measure.short_name] = [
                        measure.apply(r, g) for r, g in zip(
                            row['reconstructions'],
                            row['test_data'].ground_truth)]


    def plot_reconstruction(self,step, initial_slice, task_ind,test_ind=-1, sub_task_ind=0 ,
                            plot_ground_truth=True, **kwargs):
        """Plot the reconstruction at the specified index.
        Supports only 1d and 2d reconstructions.

        Parameters
        ----------
        Step : str
            (train, validation or test)
        initial_slice: int
            index of the minor slice
        task_ind : int
            Index of the task.
        sub_task_ind : int, optional
            Index of the sub-task (default ``0``).
        test_ind : sequence of int or int, optional
            Index in test data. If ``-1``, plot all reconstructions (the
            default).
        plot_ground_truth : bool, optional
            Whether to show the ground truth next to the reconstruction.
            The default is ``True``.
        kwargs : dict
            Keyword arguments that are passed to
            :func:`~dival.util.plot.plot_image` if the reconstruction is 2d.

        Returns
        -------
        ax_list : list of :class:`np.ndarray` of :class:`matplotlib.axes.Axes`
            The axes in which the reconstructions and eventually the ground
            truth were plotted.
        """
        row = self.results.loc[task_ind, sub_task_ind]
        test_data = row.at['test_data']
        reconstructor = row.at['reconstructor']
        ax_list = []
        if isinstance(test_ind, int):
            if test_ind == -1:
                test_ind = range(len(test_data))
        print(test_ind)
        for i in test_ind:
            
            if (i%100 == 0):
                print ('index:'+ str(i))
                
            reconstruction = row.at['reconstructions'][i]
            ground_truth = test_data.ground_truth[i]
            if reconstruction is None:
                raise ValueError('reconstruction is `None`')
            if reconstruction.asarray().ndim > 2:
                print('only 1d and 2d reconstructions can be plotted')
                return
            if reconstruction.asarray().ndim == 1:
                x = reconstruction.space.points()
                _, ax = plt.subplots()
                ax.plot(x, reconstruction, label=reconstructor.name)
                if plot_ground_truth:
                    ax.plot(x, ground_truth, label='ground truth')
                #ax.legend()
                #ax.set_title(title)
                ax = np.array(ax)
            elif reconstruction.asarray().ndim == 2:
                if plot_ground_truth:
                    _, ax = plot_images([reconstruction],
                                        **kwargs)
                    ax.figure.savefig(step+'/'+ str(i + initial_slice) + '.png' ,bbox_inches = 'tight', pad_inches = 0)
                    ax.figure.clf()
                    _, ax_gt = plot_images([ground_truth],
                                        **kwargs)
                    ax_gt.figure.savefig(step+'_gt/'+ str(i + initial_slice) +'.png' ,bbox_inches = 'tight', pad_inches = 0)
                    ax_gt.figure.clf()
                else:
                    _, ax = plot_images([reconstruction], **kwargs)
                    ax.figure.savefig(step+'/'+ str(i + initial_slice) + '.png' ,bbox_inches = 'tight', pad_inches = 0)
                    ax.figure.clf()

            plt.close('all')
            #ax_list.append([ax,ax_gt])
            
        return ax_list

    def plot_all_reconstructions(self,step,initial_slice,test_ind=-1, **kwargs):
        """Plot all reconstructions.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments that are forwarded to
            :meth:`plot_reconstruction`.

        Returns
        -------
        ax : :class:`np.ndarray` of :class:`matplotlib.axes.Axes`
            The axes the reconstructions were plotted in.
        """
        ax = []
        
        for i, j in self.results.index:
            if (i == 0):
                ax_ = self.plot_reconstruction( step, initial_slice, task_ind=i, test_ind=-1, sub_task_ind=j, 
                                               **kwargs)
                ax.append(ax_)

        return np.vstack(ax)

    def plot_convergence(self, task_ind, sub_task_ind=0, measures=None,
                         fig_size=None, gridspec_kw=None):
        """
        Plot measure values for saved iterates.

        This shows the convergence behavior with respect to the measures.

        Parameters
        ----------
        task_ind : int
            Index of the task.
        sub_task_ind : int, optional
            Index of the sub-task (default ``0``).
        measures : [list of ] :class:`.Measure`, optional
            Measures to apply. Each measure is plotted in a subplot.
            If `None` is passed, all measures in ``result['measure_values']``
            are used.

        Returns
        -------
        ax : :class:`np.ndarray` of :class:`matplotlib.axes.Axes`
            The axes the measure values were plotted in.
        """
        row = self.results.loc[task_ind, sub_task_ind]
        iterates_measure_values = row['misc'].get('iterates_measure_values')
        if not iterates_measure_values:
            iterates = row['misc'].get('iterates')
            if not iterates:
                raise ValueError(
                    "no 'iterates_measure_values' or 'iterates' in results "
                    "of task {}{}".format(
                        task_ind, '.{}'.format(sub_task_ind) if
                        len(self.results.loc[task_ind]) > 1 else ''))
        if measures is None:
            measures = row['measure_values'].keys()
        elif isinstance(measures, Measure):
            measures = [measures]
        fig, ax = plt.subplots(len(measures), 1, gridspec_kw=gridspec_kw)
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])
        if fig_size is not None:
            fig.set_size_inches(fig_size)
        fig.suptitle('convergence of {}'.format(row['reconstructor'].name))
        for measure, ax_ in zip(measures, ax.flat):
            if isinstance(measure, str):
                measure = Measure.get_by_short_name(measure)
            if iterates_measure_values:
                errors = np.mean([iters_mvs[measure.short_name] for iters_mvs
                                  in iterates_measure_values], axis=0)
            else:
                ground_truth = row['test_data'].ground_truth
                errors = np.mean([[measure.apply(x, g) for x in iters] for
                                  iters, g in zip(iterates, ground_truth)],
                                 axis=0)
            ax_.plot(errors)
            ax_.set_title(measure.short_name)
        return ax

    def plot_performance(self, measure, reconstructors=None, test_data=None,
                         weighted_average=False, **kwargs):
        """
        Plot average measure values for different reconstructors.
        The values have to be computed previously, e.g. by
        :meth:`apply_measures`.

        The average is computed over all rows of :attr:`results` with the
        specified `test_data` that store the requested `measure` value.

        Note that for tasks with multiple sub-tasks, all of them are used when
        computing the average (i.e., the measure values for all hyper parameter
        choices are averaged).

        Parameters
        ----------
        measure : :class:`.Measure` or str
            The measure to plot (or its :attr:`~.Measure.short_name`).
        reconstructors : sequence of :class:`.Reconstructor`, optional
            The reconstructors to compare. If `None` (default), all
            reconstructors that are found in the results are compared.
        test_data : [sequence of ] :class:`.DataPairs`, optional
            Test data to take into account for computing the mean value.
            By default, all test data is used.
        weighted_average : bool, optional
            Whether to weight the rows according to the number of pairs in
            their test data.
            Default: ``False``, i.e. all rows are weighted equally.
            If ``True``, all test data pairs are weighted equally.

        Returns
        -------
        ax : `matplotlib.axes.Axes`
            The axes the performance was plotted in.
        """
        if not isinstance(measure, Measure):
            measure = Measure.get_by_short_name(measure)
        if reconstructors is None:
            reconstructors = self.results['reconstructor'].unique()
        if isinstance(test_data, DataPairs):
            test_data = [test_data]
        mask = [measure.short_name in row['measure_values'].keys() and
                row['reconstructor'] in reconstructors and
                (test_data is None or row['test_data'] in test_data)
                for _, row in self.results.iterrows()]
        rows = self.results[mask]
        v = []
        for reconstructor in reconstructors:
            r_rows = rows[rows['reconstructor'] == reconstructor]
            values = [mvs[measure.short_name] for mvs in
                      r_rows['measure_values']]
            weights = None
            if weighted_average:
                weights = [len(test_data.observations) for test_data in
                           r_rows['test_data']]
            v.append(np.average(values, weights=weights))
        fig, ax = plt.subplots(**kwargs)
        ax.bar(range(len(v)), v)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels([r.name for r in reconstructors], rotation=30)
        ax.set_title('{measure_name}'.format(measure_name=measure.name))
        return ax

    def to_string(self, max_colwidth=70, formatters=None, hide_columns=None,
                  show_columns=None, **kwargs):
        """Convert to string. Used by :meth:`__str__`.

        Parameters
        ----------
        max_colwidth : int, optional
            Maximum width of the columns, c.f. the option
            ``'display.max_colwidth'`` of pandas.
        formatters : dict of functions, optional
            Custom formatter functions for the columns, passed to
            :meth:`results.to_string <pandas.DataFrame.to_string>`.
        hide_columns : list of str, optional
            Columns to hide. Default: ``['reconstructions', 'misc']``.
        show_columns : list of str, optional
            Columns to show. Overrides `hide_columns`.
        kwargs : dict
            Keyword arguments passed to
            :meth:`results.to_string <pandas.DataFrame.to_string>`.

        Returns
        -------
        string : str
            The string.
        """

        def measure_values_formatter(measure_values):
            means = ['{}: {:.4g}'.format(k, np.mean(v)) for k, v in
                     measure_values.items()]
            return 'mean: {{{}}}'.format(', '.join(means))

        def name_or_repr_formatter(x):
            return x.name or x.__repr__()

        formatters_ = {}
        formatters_['measure_values'] = measure_values_formatter
        formatters_['test_data'] = name_or_repr_formatter
        formatters_['reconstructor'] = name_or_repr_formatter
        if formatters is not None:
            formatters_.update(formatters)
        if hide_columns is None:
            hide_columns = ['reconstructions', 'misc']
        if show_columns is None:
            show_columns = []
        columns = [c for c in self.results.columns if c not in hide_columns or
                   c in show_columns]
        with pd.option_context('display.max_colwidth', max_colwidth):
            return "ResultTable(results=\n{}\n)".format(
                self.results.to_string(formatters=formatters_, columns=columns,
                                       **kwargs))

    def print_summary(self):
        """Prints a summary of the results.
        """
        print('ResultTable with {:d} tasks.'.format(
            len(self.results.index.levels[0])))
        if len(self.results.index.levels[1]) > 1:
            print('Total count of sub-tasks: {}'.format(len(self.results)))
        test_data_list = pd.unique(self.results['test_data'])
        if len(test_data_list) == 1:
            print('Test data: {}'.format(test_data_list[0]))

    def __repr__(self):
        return "ResultTable(results=\n{results})".format(results=self.results)

    def __str__(self):
        return self.to_string()