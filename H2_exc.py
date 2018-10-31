from collections import OrderedDict
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from scipy import integrate
import sys
sys.path.append('/home/toksovogo/science/codes/python')
sys.path.append('/science/python')
from spectro.a_unc import a
from spectro.sviewer.utils import Timer

def column(matrix, i):
    if i == 0 or (isinstance(i, str) and i[0] == 'v'):
        return np.asarray([row.val for row in matrix])
    if i == 1 or (isinstance(i, str) and i[0] == 'p'):
        return np.asarray([row.plus for row in matrix])
    if i == 2 or (isinstance(i, str) and i[0] == 'm'):
        return np.asarray([row.minus for row in matrix])

H2_energy = np.genfromtxt('energy_X_H2.dat', dtype=[('nu', 'i2'), ('j', 'i2'), ('e', 'f8')],
                          unpack=True, skip_header=3, comments='#')
H2energy = np.zeros([max(H2_energy['nu']) + 1, max(H2_energy['j']) + 1])
for e in H2_energy:
    H2energy[e[0], e[1]] = e[2]

stat = [(2 * i + 1) * ((i % 2) * 2 + 1) for i in range(12)]

spcode = {'H': 'n_h', 'H2': 'n_h2',
          'H2j0': 'pop_h2_v0_j0', 'H2j1': 'pop_h2_v0_j1', 'H2j2': 'pop_h2_v0_j2', 'H2j3': 'pop_h2_v0_j3',
          'H2j4': 'pop_h2_v0_j4', 'H2j5': 'pop_h2_v0_j5', 'H2j6': 'pop_h2_v0_j6', 'H2j7': 'pop_h2_v0_j7',
          'C': 'n_c', 'C+': 'n_cp', 'CO': 'n_co',
          'NH2': 'cd_prof_h2',  'NH2j0': 'cd_lev_prof_h2_v0_j0', 'NH2j1': 'cd_lev_prof_h2_v0_j1','NH2j2': 'cd_lev_prof_h2_v0_j2','NH2j3': 'cd_lev_prof_h2_v0_j3','NH2j4': 'cd_lev_prof_h2_v0_j4','NH2j5': 'cd_lev_prof_h2_v0_j5',
          'H2_dest_rate': 'h2_dest_rate_ph', 'H2_form_rate_er': 'h2_form_rate_er','H2_form_rate_lh': 'h2_form_rate_lh','H2_photo_dest_prob': 'photo_prob___h2_photon_gives_h_h',
          'cool_tot': 'coolrate_tot', 'cool_cp': 'coolrate_cp', 'heat_tot': 'heatrate_tot', 'heat_phel': 'heatrate_pe'
         }

class plot(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29))
        self.initstatus()
        self.vb = self.getViewBox()

    def initstatus(self):
        self.s_status = False
        self.selected_point = None

    def set_data(self, data=None):
        if data is None:
            try:
                self.vb.removeItem(self.d)
            except:
                pass
        else:
            self.data = data
            self.points = pg.ScatterPlotItem(self.data[0], self.data[1], symbol='o', pen={'color': 0.8, 'width': 1}, brush=pg.mkBrush(100, 100, 200))
            self.vb.addItem(self.points)

    def mousePressEvent(self, event):
        super(plot, self).mousePressEvent(event)
        print('KEY PRESSED')
        if event.button() == Qt.LeftButton:
            if self.s_status:
                self.mousePoint = self.vb.mapSceneToView(event.pos())
                r = self.vb.viewRange()
                self.ind = np.argmin(((self.mousePoint.x() - self.data[0]) / (r[0][1] - r[0][0]))**2   + ((self.mousePoint.y() - self.data[1]) / (r[1][1] - r[1][0]))**2)
                if self.selected_point is not None:
                    self.vb.removeItem(self.selected_point)
                self.selected_point = pg.ScatterPlotItem(x=[self.data[0][self.ind]], y=[self.data[1][self.ind]], symbol='o', size=15,
                                                        pen={'color': 0.8, 'width': 1}, brush=pg.mkBrush(230, 100, 10))
                self.vb.addItem(self.selected_point)


    def keyPressEvent(self, event):
        super(plot, self).keyPressEvent(event)
        key = event.key()

        if not event.isAutoRepeat():
            if event.key() == Qt.Key_S:
                self.s_status = True

    def keyReleaseEvent(self, event):
        super(plot, self).keyReleaseEvent(event)
        key = event.key()

        if not event.isAutoRepeat():
            if event.key() == Qt.Key_S:
                self.s_status = False


class model():
    def __init__(self, folder='', name=None, filename=None, species=[], show_summary=True, show_meta=False):
        self.folder = folder
        self.sp = {}
        self.species = species
        self.filename = filename
        self.initpardict()
        if filename is not None:
            if name is None:
                name = filename.replace('.hdf5', '')
            self.name = name
            self.read(show_summary=show_summary, show_meta=show_meta)

    def initpardict(self):
        self.pardict = {}
        self.pardict['metal'] = ('Parameters/Parameters', 13, float)
        self.pardict['radm_ini'] = ('Parameters/Parameters', 5, float)
        self.pardict['proton_density_input'] = ('Parameters/Parameters', 3, float)
        self.pardict['distance'] = ('Local quantities/Positions', 2, float)
        self.pardict['av'] = ('Local quantities/Positions', 1, float)
        self.pardict['tgas'] = ('Local quantities/Gas state', 2, float)
        self.pardict['pgas'] = ('Local quantities/Gas state', 3, float)
        self.pardict['ntot'] = ('Local quantities/Gas state', 1, float)
        for n, ind in zip(['n_h', 'n_h2', 'n_c', 'n_cp', 'n_co'], [0, 2, 5, 93, 44]):
            self.pardict[n] = ('Local quantities/Densities/Densities', ind, float)
        for i in range(8):
            self.pardict['pop_h2_v0_j'+str(i)] = ('Local quantities/Auxiliary/Excitation/Level densities', 9+i, float)

    def read(self, show_meta=False, show_summary=True, fast=True):
        """
        Read model data from hdf5 file

        :param
            -  show_meta           :  if True, show Metadata table
            -  show_summary        :  if True, show summary

        :return: None
        """
        self.file = h5py.File(self.folder + self.filename, 'r')

        self.fastread = fast

        # >>> model input parameters
        self.me = self.par('metal')
        self.P = self.par('gas_pressure_input')
        self.n0 = self.par('proton_density_input')
        self.uv = self.par('radm_ini')

        # >>> profile of physical quantities
        self.x = self.par('distance')

        self.h2 = self.par('cd_prof_h2')
        self.av = self.par('av')
        self.tgas = self.par('tgas')
        self.pgas = self.par('pgas')
        self.n = self.par('ntot')
        self.nH = self.par('protdens')
        self.uv_flux = self.par('uv_flux')
        self.uv_dens = self.par('uv_dens')

        self.readspecies()

        if 0:
            self.plot_phys_cond(pars=['tgas', 'n'], parx='av', logx=False)
            #self.plot_profiles()

        if show_meta:
            self.showMetadata()

        self.file.close()

        if show_summary:
            self.showSummary()

    def par(self, par=None):
        """
        Read parameter or array from the hdf5 file, specified by Metadata name

        :param:
            -  par          :  the name of the parameter to read

        :return: x
            - x             :  corresponding data (string, number, array) correspond to the parameter
        """
        meta = self.file['Metadata/Metadata']
        if par is not None:
            if self.fastread and par in self.pardict:
                attr, ind, typ = self.pardict[par]
            else:
                ind = np.where(meta[:, 3] == par.encode())[0]
                if len(ind) > 0:
                    attr = meta[ind, 0][0].decode() + '/' + meta[ind, 1][0].decode()
                    typ = {'string': str, 'real': float, 'integer': int}[meta[ind, 5][0].decode()]
                    ind = int(meta[ind, 2][0].decode())
                else:
                    return None
        x = self.file[attr][:, ind]
        if len(x) == 1:
            return typ(x[0].decode())
        else:
            return x

    def showMetadata(self):
        """
        Show Metadata information in the table
        """

        self.w = pg.TableWidget()
        self.w.show()
        self.w.resize(500, 900)
        self.w.setData(self.file['Metadata/Metadata'][:])

    def readspecies(self, species=None):
        """
        Read the profiles of the species

        :param
            -  species       : the list of the names of the species to read from the file

        :return: None

        """
        if species is None:
            species = self.species

        for s in species:
            self.sp[s] = self.par(spcode[s])

        self.species = species

    def showSummary(self, pars=['z', 'P', 'uv']):
        print('model: ' + self.name)
        for p in pars:
            print(p, ' : ', getattr(self, p))
            #print("{0:s} : {1:.2f}".format(p, getattr(self, p)))

    def plot_phys_cond(self, pars=['tgas', 'n', 'av'], logx=True, ax=None, legend=True, parx='x'):
        """
        Plot the physical parameters in the model

        :parameters:
            -  pars         :  list of the parameters to plot
            -  logx         :  if True x in log10
            -  ax           :  axes object to plot
            -  legend       :  show Legend

        :return: ax
            -  ax           :  axes object
        """

        if parx == 'av':
            xlabel = 'Av'
        elif parx == 'x':
            xlabel = 'log(Distance), cm'
        elif parx == 'h2':
            xlabel = 'log(NH2), cm-2'

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        if logx:
            mask = getattr(self, parx) > 0
            x = np.log10(getattr(self, parx)[mask])
        else:
            mask = getattr(self, parx) > -1
            x = getattr(self, parx)[mask]
        ax.set_xlim([x[0], x[-1]])
        ax.set_xlabel(xlabel)

        lines = []
        for i, p in enumerate(pars):
            if i == 0:
                axi = ax
            else:
                axi = ax.twinx()

            if p in ['tgas', 'n', 'av', 'Pgas']:
                y = getattr(self, p)[mask]
            else:
                if 'N_' in p:
                    y = np.log10(integrate.cumtrapz(self.sp[p.replace('N_', '')][mask], 10**x, initial=0))
                else:
                    y = self.sp[p][mask]

            color = plt.cm.tab10(i / 10)
            axi.set_ylabel(p, color=color)
            line, = axi.plot(x, y, color=color, label=p)
            lines.append(line)

            if i > 0:
                axi.spines['right'].set_position(('outward', 60*(i-1)))
                axi.set_frame_on(True)
                axi.patch.set_visible(False)

            for t in axi.get_yticklabels():
                t.set_color(color)

        if legend:
            #ax.legend()
            ax.legend(handles=lines, loc='best')

        return ax

    def plot_profiles(self, species=None, logx=False, logy=False, label=True, ax=None, legend=True, ls='-', lw=1, parx='av'):
        """
        Plot the profiles of the species

        :param:
            -  species       :  list of the species to plot
            -  ax            :  axis object to plot in, if None, then figure is created
            -  legend        :  show legend
            -  ls            :  linestyles
            -  lw            :  linewidth
            -  logx          :  log of x axis
            -  label         :  set label of x axis

        :return: ax
            -  ax            :  axis object
        """
        if species is None:
            species = self.species

        if parx == 'av':
            xlabel = 'Av'
        elif parx == 'x':
            xlabel = 'log(Distance), cm'
        elif parx == 'h2':
            xlabel = 'log(NH2), cm-2'

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        if logx:
            mask = getattr(self, parx) > 0
            x = np.log10(getattr(self, parx)[mask])
        else:
            mask = getattr(self, parx) > -1
            x = getattr(self, parx)[mask]
        ax.set_xlim([x[0], x[-1]])
        ax.set_xlabel(xlabel)


        if ax is None:
            fig, ax = plt.subplots()

#        for s in species:
#            ax.plot(np.log10(self.x[1:]), np.log10(self.sp[s][1:]), '-', label=s, lw=lw)
        for s in species:
            if logy:
                ax.plot(x[mask], np.log10(self.sp[s][mask]), ls=ls, label=s, lw=lw, linewidth=2.0)
            else:
                ax.plot(x[mask], self.sp[s][mask], ls=ls, label=s, lw=lw, linewidth=2.0)

        if label:
            #ax.set_xlim([x[0], x[-1]])
            ax.set_xlabel(xlabel,fontsize=20)
            ax.set_ylabel(species[0],fontsize=20)
            ax.legend(loc='upper left')

        if legend:
            #ax.legend(labels=[self.name])
            ax.legend()

        return ax

    def calc_cols(self, species=[], logN=None, sides=2):
        """
        Calculate column densities for given species

        :param:
            -  species       :  list of the species to plot
            -  logN          :  column density threshold, dictionary with species and logN value
            -  side          :  make calculation to be one or two sided

        :return: sp
            -  sp            :  dictionary of the column densities by species
        """

        if logN is not None:
            logN[list(logN.keys())[0]] -= np.log10(sides)
            self.set_mask(logN=logN)

        cols = {}
        for s in species:
            cols[s] = np.log10(np.trapz(self.sp[s][self.mask], x=self.x[self.mask]) * sides)

        self.cols = cols

        return self.cols

    def set_mask(self, logN={'H': None}):
        """
        Calculate mask for a given threshold

        :param:
            -  logN          :  column density threshold

        :return: None
        """
        cols = np.insert(np.log10(integrate.cumtrapz(self.sp[list(logN.keys())[0]], x=self.x)), 0, 0)
        if logN is not None:
            self.mask = cols < logN[list(logN.keys())[0]]
        else:
            self.mask = cols > -1
        #return np.searchsorted(cols, value)
        #print('av_max:', self.av[self.mask][-1])

    def lnLike(self, species={}, syst=0, verbose=False):
        lnL = 0
        if verbose:
            self.showSummary()
        for k, v in species.items():
            v1 = v
            v1 *= a(0, syst, syst, 'l')
            if verbose:
                print(self.cols[k], v1.log(), v1.lnL(self.cols[k]))
            if v.type == 'm':
                lnL += v1.lnL(self.cols[k])

        self.lnL = lnL
        return self.lnL

class H2_exc():
    def __init__(self, folder=''):
        self.folder = folder
        self.models = {}
        self.species = ['H', 'H2', 'C', 'C+', 'CO', 'H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5', 'H2j6', 'H2j7',
                        'NH2','NH2j5', 'NH2j4','NH2j3', 'NH2j2','NH2j1', 'NH2j0',
                        'H2_dest_rate', 'H2_form_rate_er', 'H2_form_rate_lh', 'H2_photo_dest_prob']
        self.readH2database()

    def readH2database(self):
        import sys
        sys.path.append('/home/toksovogo/science/codes/python/3.5/')
        import H2_summary

        self.H2 = H2_summary.load_QSO()

    def readmodel(self, filename=None, show_summary=False):
        """
        Read one model by filename
        :param:
            -  filename             :  filename contains the model
            -  print_summary        :  if True, print summary for each model
        """
        if filename is not None:
            m = model(folder=self.folder, filename=filename, species=self.species, show_summary=False)
            self.models[m.name] = m
            self.current = m.name

            if show_summary:
                m.showSummary()

    def readfolder(self, verbose=False):
        """
        Read list of models from the folder
        """
        for f in os.listdir(self.folder):
            if f.endswith('.hdf5'):
                self.readmodel(f, show_summary=verbose)

    def setgrid(self, pars=[], fixed={}, show=True):
        """
        Show and mask models for grid of specified parameters
        :param:
            -  pars           :  list of parameters in the grid, e.g. pars=['uv', 'P']
            -  fixed          :  dict of parameters to be fixed, e.g. fixed={'z': 0.1}
            -  show           :  if true show the plot for the grid
        :return: mask
            -  mask           :  list of names of the models in the grid
        """
        self.grid = {p: [] for p in pars}
        self.mask = []

        for name, model in self.models.items():
            for k, v in fixed.items():
                if getattr(model, k) != v:
                    break
            else:
                for p in pars:
                    self.grid[p].append(getattr(model, p))
                self.mask.append(name)

        #print(self.grid)
        if show and len(pars) == 2:
            fig, ax = plt.subplots()
            for v1, v2 in zip(self.grid[pars[0]], self.grid[pars[1]]):
                ax.scatter(v1, v2, 100, c='orangered')
            ax.set_xscale("log", nonposy='clip')
            ax.set_yscale("log", nonposy='clip')
            ax.set_xlabel(pars[0])
            ax.set_ylabel(pars[1])

        if show and len(pars) == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for v1, v2, v3 in zip(self.grid[pars[0]], self.grid[pars[1]], self.grid[pars[2]]):
                ax.scatter(v1, v2, v3, c='orangered')
            ax.set_xlabel(pars[0])
            ax.set_ylabel(pars[1])
            ax.set_zlabel(pars[2])

        return self.mask

    def comp(self, object):
        """
        Return componet object from self.H2
        :param:
            -  object         :  object name.
                                    Examples: '0643' - will search for the 0643 im quasar names. Return first component.
                                              '0643_1' - will search for the 0643 im quasar names. Return second component
        :return: q
            -  q              :  qso.comp object (see file H2_summary.py how to retrieve data (e.g. column densities) from it
        """
        qso = self.H2.get(object.split('_')[0])
        if len(object.split('_')) > 1:
            q = qso.comp[int(object.split('_')[1])]
        else:
            q = qso.comp[0]

        return q

    def listofmodels(self, models=[]):
        """
        Return list of models

        :param:
            -  models         :  names of the models, can be list or string for individual model

        :return: models
            -  models         :  list of models

        """
        if isinstance(models, str):
            if models == 'current':
                models = [self.models[self.current]]
            elif models == 'all':
                models = list(self.models.values())
            else:
                models = [self.models[models]]
        elif isinstance(models, list):
            if len(models) == 0:
                models = list(self.models.values())
            else:
                models = [self.models[m] for m in models]

        return models

    def compare(self, object='', models='current', syst=0.0):
        """
        Calculate the column densities of H2 rotational levels for the list of models given the total H2 column density.
        and also log of likelihood

        :param:
            -  object            :  object name
            -  models            :  names of the models, can be list or string
            -  syst              :  add systematic uncertainty to the calculation of the likelihood

        :return: None
            column densities are stored in the dictionary <cols> attribute for each model
            log of likelihood value is stored in <lnL> attribute
        """

        q = self.comp(object)

        for model in self.listofmodels(models):
            #print(model)
            species = OrderedDict([(s, q.e[s].col) for s in q.e.keys() if 'H2j' in s])
            model.calc_cols(species.keys(), logN={'H2': q.e['H2'].col.val})
            model.lnLike(species, syst=syst)

    def comparegrid(self, object='0643', pars=[], fixed={}, syst=0.0, plot=True, show_best=True):

        self.setgrid(pars=pars, fixed=fixed, show=False)
        self.compare(object, models=self.mask, syst=syst)
        self.grid['lnL'] = np.asarray([self.models[m].lnL for m in self.mask])

        if plot:
            if len(pars) == 1:
                x = np.asarray(self.grid[list(self.grid.keys())[0]])
                inds = np.argsort(x)
                fig, ax = plt.subplots()
                ax.scatter(x[inds], self.grid['lnL'][inds], 100, c='orangered')
                self.plot = plot(self)
                self.plot.set_data([x[inds], self.grid['lnL'][inds]])
                self.plot.show()

            if len(pars) == 2:
                fig, ax = plt.subplots()
                for v1, v2, l in zip(self.grid[pars[0]], self.grid[pars[1]], self.grid['lnL']):
                    ax.scatter(v1, v2, 0)
                    ax.text(v1, v2, '{:.1f}'.format(l), size=20)

            if show_best:
                imax = np.argmax(lnL)
                ax = self.plot_objects(objects=object)
                self.plot_models(ax=ax, models=self.mask[imax])

    def plot_objects(self, objects=[], species=[], ax=None, plotstyle='scatter', legend=False):
        """
        Plot object from the data

        :param:
            -  objects              :  names of the object to plot
            -  species              :  names of the species to plot
            -  ax                   :  axes object, where to plot. If None, then it will be created
            -  plotstyle            :  style of plotting. Can be 'scatter' or 'lines'
            -  legend               :  show legend

        :return: ax
            -  ax                   :  axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        if not isinstance(objects, list):
            objects = [objects]
        for o in objects:
            q = self.comp(o)
            if species is None or len(species) == 0:
                sp = [s for s in q.e.keys() if 'H2j' in s]
            j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
            x = [H2energy[0, i] for i in j]
            y = [q.e['H2j' + str(i)].col / stat[i] for i in j]
            typ = [q.e['H2j' + str(i)].col.type for i in j]
            if len(y) > 0:
                color = plt.cm.tab10(objects.index(o) / 10)
                if plotstyle == 'line':
                    ax.plot(x, column(y, 'v'), marker='o', ls='-', lw=2, color=color, label=o)
                for k in range(len(y)):
                    if typ[k] == 'm':
                        ax.errorbar([x[k]], [column(y, 0)[k]], yerr=[[column(y, 1)[k]], [column(y, 2)[k]]],
                                    fmt='o', lw=0, elinewidth=1, color=color, label=o)
                    if typ[k] == 'u':
                        ax.errorbar([x[k]], [column(y, 0)[k]], yerr=[[0.4], [0.4]],
                                    fmt='o', uplims=0.2, lw=1, elinewidth=1, color=color)

        if legend:
            handles, labs = ax.get_legend_handles_labels()
            labels = np.unique(labs)
            handles = [handles[np.where(np.asarray(labs) == l)[0][0]] for l in labels]
            ax.legend(handles, labels, loc='best')

        return ax

    def plot_models(self, ax=None, models='current', logN=None, species='<7', legend=False):
        """
        Plot excitation for specified models

        :param:
            -  ax                :  axes object, where to plot. If None, it will be created
            -  models            :  names of the models to plot
            -  logN              :  total H2 column density
            -  species           :  list of rotational levels to plot, can be string as '<N', where N is the number of limited rotational level
            -  legend            :  if True, then plot legend

        :return: ax
            -  ax                :  axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        if isinstance(species, str):
            species = ['H2j' + str(i) for i in range(int(species[1:]))]

        if logN is not None:
            for m in self.listofmodels(models):
                print(m)
                m.calc_cols(species, logN={'H2': logN})

        for ind, m in enumerate(self.listofmodels(models)):
            j = np.sort([int(s[3:]) for s in m.cols.keys()])
            x = [H2energy[0, i] for i in j]
            mod = [m.cols['H2j'+str(i)] - np.log10(stat[i]) for i in j]

            if len(mod) > 0:
                color = plt.cm.tab10(ind/10)
                ax.plot(x, mod, marker='', ls='--', lw=1, color=color, label=m.name, zorder=0, linewidth=2.0)
                ax.legend()

        if legend:
            ax.legend(loc='best')

        return ax

    def compare_models(self, speciesname=['H'], ax=None, models='current', physcondname=False, logy=False, parx='av'):
        """
        Plot comparison of certain quantities for specified models

        :param:
            -  ax                :  axes object, where to plot. If None, it will be created
            -  models            :  names of the models to plot
            -  speciesname       :  name of plotted from list ['H', 'H2', 'C', 'C+', 'CO', 'H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5', 'H2j6', 'H2j7']

        :return: ax
            -  ax                :  axes object
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        if 1:
            legend_m = []
            for ind, m in enumerate(self.listofmodels(models)):
                m.plot_profiles(species=speciesname, ax=ax, logy=logy,parx=parx)
                legend_m.append(m.name[0:40])
            ax.legend(labels=legend_m, fontsize=20)
            ax.set_title(speciesname)

        if physcondname:
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            legend_m = []
            for ind, m in enumerate(self.listofmodels(models)):
                m.plot_phys_cond(pars=physcondname, logx=False, ax=ax2, parx=parx)
                legend_m.append(m.name[0:40])
            ax2.legend(labels=legend_m, fontsize=20)

        if physcondname:
            return ax, ax2
        else:
            return ax

    def show_model(self, models='current',  ax=None, speciesname=['H'],  physcond_show=True, logy=False):
        """
        Plot excitation for specified models

        :param:
            -  ax                :  axes object, where to plot. If None, it will be created
            -  models            :  names of the models to plot
            -  speciesname       :  name of plotted from list ['H', 'H2', 'C', 'C+', 'CO', 'H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5', 'H2j6', 'H2j7']

        :return: ax
            -  ax                :  axes object
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        if 1:
            #legend_m = []
            for ind, m in enumerate(self.listofmodels(models)):
                for s in speciesname:
                    m.plot_profiles(species=[s], ax=ax, logy=logy)

            ax.set_title(m.name[0:30])

        if physcond_show:
            if 1:
                fig2, ax2 = plt.subplots(figsize=(12, 8))
                legend_m = []
                for ind, m in enumerate(self.listofmodels(models)):
                    m.plot_phys_cond(pars=['n', 'tgas'], parx='av', logx=False, ax=ax2)
                    legend_m.append(m.name[8:19])
                ax2.legend(fontsize=20)
                ax2.set_title(m.name[0:30])

        if physcond_show is False:
            return ax
        else:
            return ax,  ax2

    def red_hdf_file(self, filename=None):
        self.file = h5py.File(self.folder + filename, 'r+')
        meta = self.file['Metadata/Metadata']
        ind = np.where(meta[:, 3] == 'metal'.encode())[0]
        attr = meta[ind, 0][0].decode() + '/' + meta[ind, 1][0].decode()
        data = self.file[attr]
        d = data[0, int(meta[ind, 2][0].decode())]
        data[0, int(meta[ind, 2][0].decode())] = d[0:3] + b'60' + d[5:]
        self.file.close()


    def best(self, object='', models='all', syst=0.0):

        models = self.listofmodels(models)

        self.compare(object=object, models=[m.name for m in models], syst=syst)

        return models[np.argmax([m.lnL for m in models])].name

if __name__ == '__main__':
    import sys

    app = QtGui.QApplication([])

    fig, ax = plt.subplots()
    H2 = H2_exc(folder='data/')
    if 0:
        H2.readfolder()
        H2.plot_objects(objects='2123', ax=ax)
        name = H2.best(object='2123', syst=0.1)
        print(H2.models[name].uv)
        if 1:
            H2.plot_models(ax=ax, models='all')
            H2.compare_models(speciesname=['NH2'], models='all', physcondname=['tgas'], logy=True,
                              parx='x')  # physcondname=['tgas','n'],
            H2.compare_models(speciesname=['NH2'], models='all', logy=True, parx='av')
            # H2.compare_models(speciesname=['H2_photo_dest_prob'], models='all', logy=True, parx='x')
            # H2.compare_models(speciesname=['NH2'], models='all', logy=True)
            # H2.compare_models(speciesname=['NH2'], models='all', logy=True, parx='x')
            # H2.compare_models(speciesname=['H2'], models='all', logy=True, parx='x')
        else:
            H2.plot_models(ax=ax, models=name)

    H2.plot_objects(objects=H2.H2.all())
    if 1:
        m = model(folder='data/', filename='h2uv_uv12_av0_05_z0_16_n3e1_s_25.hdf5', show_meta=True, species=['H', 'H2', 'H2j0', 'H2j1'])
        m.plot_phys_cond(pars=['tgas', 'n', 'av', 'N_H2'])
    if 0:
        H2.readfolder()
        #H2.plot_objects(objects=['0643', '0843'], ax=ax)
        if 0:
            name = H2.best(object='0643', syst=0.1)
            print(H2.models[name].lnL, H2.models[name].uv, H2.models[name].z, H2.models[name].P)
        if 0:
            H2.compare('J0643', models='all', syst=0.1)
            H2.plot_models(ax=ax, models='all')
        if 0:
            H2.plot_models(ax=ax, models=name)
        if 0:
            #H2.setgrid(pars=['uv', 'n0'], fixed={'z': 0.160})
            H2.comparegrid('B0528_1', pars=['uv', 'n0'], fixed={'me': 0.160}, syst=0.1)
            #H2.comparegrid('B0107_1', pars=['uv', 'n0'], fixed={'z': 0.160}, syst=0.1)
        if 1:
            H2.comparegrid('J0643', pars=['uv', 'P'], fixed={'me': 0.160}, syst=0.1)
    plt.tight_layout()
    plt.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



