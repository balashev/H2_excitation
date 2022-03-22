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
sys.path.append('/media/serj/OS/science/python')
sys.path.append('/science/python')
sys.path.append('/science/spectro')
from spectro.a_unc import a
from spectro.sviewer.utils import Timer
import warnings

def column(matrix, i):
    if i == 0 or (isinstance(i, str) and i[0] == 'v'):
        return np.asarray([row.val for row in matrix])
    if i == 1 or (isinstance(i, str) and i[0] == 'p'):
        return np.asarray([row.plus for row in matrix])
    if i == 2 or (isinstance(i, str) and i[0] == 'm'):
        return np.asarray([row.minus for row in matrix])

H2_energy = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + r'/energy_X_H2.dat', dtype=[('nu', 'i2'), ('j', 'i2'), ('e', 'f8')],
                          skip_header=3, comments='#')
H2energy = np.zeros([max(H2_energy['nu']) + 1, max(H2_energy['j']) + 1])
for e in H2_energy:
    H2energy[e[0], e[1]] = e[2]
CIenergy = [0, 16.42, 43.41]
COenergy = [1.922529 * i * (i + 1) - 6.1206e-6 * (i * (i + 1)) ** 2 for i in range(15)]

stat_H2 = [(2 * i + 1) * ((i % 2) * 2 + 1) for i in range(12)]
stat_CI = [(2 * i + 1) for i in range(3)]
stat_CO = [(2 * i + 1) for i in range(15)]

spcode = {'H': 'n_h', 'H2': 'n_h2', 'H+': 'n_hp', 'H-': 'n_hm', 'H2+': 'n_h2p', 'H3+': 'n_h3p', 'e': 'n_electr',
          'He': 'n_he', 'He+': 'n_hep', 'HeH+': 'n_hehp',
          'D': 'n_d', 'D+': 'n_dp', 'HD': 'n_hd', 'HDj0': 'pop_hd_v0_j0', 'HDj1': 'pop_hd_v0_j1', 'HDj2': 'pop_hd_v0_j2',
          'C': 'n_c', 'C+': 'n_cp', 'CO': 'n_co', 'HCO+': 'n_hcop', 'CO+': 'n_cop', 'CO2': 'n_co2',
          'CH': 'n_ch', 'CH+': 'n_chp', 'CH2': 'n_ch2', 'CH3': 'n_ch3', 'CH4': 'n_ch4',
          'CN': 'n_cn',
          'O': 'n_o', 'O+': 'n_op', 'OH+': 'n_ohp', 'H2O+': 'n_h2op', 'H3O+': 'n_h3op', 'OH': 'n_oh', 'H2O': 'n_h2o',
          'CIj0': 'pop_c_el3p_j0', 'CIj1': 'pop_c_el3p_j1', 'CIj2': 'pop_c_el3p_j2',
          'SiI': 'n_si', 'SiII': 'n_sip', 'SiIII': 'n_sipp', 'SiIIj0': 'pop_sip_el2p_j1_2', 'SiIIj1': 'pop_sip_el2p_j3_2', 'NSiII': 'cd_prof_sip',
          'Ar+': 'n_arp', 'ArH+': 'n_arhp',
          'NHI': 'cd_prof_h', 'NH2': 'cd_prof_h2',  'NHD': 'cd_prof_hd', 'NOH': 'cd_prof_oh', 'NCO': 'cd_prof_co', 'NCN': 'cd_prof_cn',
          'NHItot': 'cd_h', 'NH2tot': 'cd_h2',  'NHDtot': 'cd_hd', 'NOHtot': 'cd_oh', 'NCOtot': 'cd_co',
          'H2_dest_rate': 'h2_dest_rate_ph', 'H2_form_rate_er': 'h2_form_rate_er','H2_form_rate_lh': 'h2_form_rate_lh','H2_photo_dest_prob': 'photo_prob___h2_photon_gives_h_h',
          'cool_tot': 'coolrate_tot', 'cool_cp': 'coolrate_cp', 'heat_tot': 'heatrate_tot', 'heat_phel': 'heatrate_pe',
          'H2_diss': 'h2_dest_rate_ph',
          'Av': 'av',
         }
H2dict = {f'H2j{i}': f'pop_h2_v0_j{i}' for i in range(12)}
NH2dict = {f'NH2j{i}': f'cd_lev_prof_h2_v0_j{i}' for i in range(12)}
COdict = {f'COj{i}': f'pop_co_v0_j{i}' for i in range(12)}
spcode = {**spcode, **H2dict, **NH2dict, **COdict}

splatex = {'H': r'$\rm H$', 'H2': r'$\rm H_2$', 'H+': r'$\rm H^+$', 'H2+': r'$\rm H_2^+$', 'H3+': r'$\rm H_3^+$',
           'O': r'$\rm O$', 'O+': r'$\rm O^+$', 'OH': r'$\rm OH$', 'OH+': r'$\rm OH^+$', 'H2O': r'$\rm H_2O$', 'H2O+': r'$\rm H_2O^+$', 'H3O+': r'$\rm H_3O^+$',
           'C': r'$\rm C$', 'C+': r'$\rm C^+$', 'CO': r'$\rm CO$',
           'He': r'$\rm He$', 'He+': r'$\rm He^+$', 'HeH+': r'$\rm HeH^+$', 'Ar+': r'$\rm Ar^+$', 'ArH+': r'$\rm ArH^+$'
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
    def __init__(self, folder='', name=None, filename=None, species=[], show_summary=True, show_meta=False, fastread=True):
        self.folder = folder
        self.sp = {}
        self.species = species
        self.filename = filename
        self.initpardict()
        self.fastread = fastread
        if filename is not None:
            if name is None:
                name = filename.replace('.hdf5', '')
            self.name = name
            self.read(show_summary=show_summary, show_meta=show_meta)

    def initpardict(self):
        self.pardict = {}
        self.pardict['codeversion'] = ('Parameters/Informations/', 0, str)
        self.pardict['metal'] = ('Parameters/Parameters', 13, float)
        self.pardict['radm_ini'] = ('Parameters/Parameters', 5, float)
        self.pardict['proton_density_input'] = ('Parameters/Parameters', 3, float)
        self.pardict['avmax'] = ('Parameters/Parameters', 1, float)
        self.pardict['distance'] = ('Local quantities/Positions', 2, float)
        self.pardict['av'] = ('Local quantities/Positions', 1, float)
        self.pardict['tgas'] = ('Local quantities/Gas state', 2, float)
        self.pardict['pgas'] = ('Local quantities/Gas state', 3, float)
        self.pardict['ntot'] = ('Local quantities/Gas state', 1, float)
        for n, ind in zip(['n_h', 'n_hp', 'n_h2', 'n_c', 'n_cp', 'n_co'], [0, 92, 2, 5, 100, 46]):
            self.pardict[n] = ('Local quantities/Densities/Densities', ind, float)
        for i in range(11):
            self.pardict[f'pop_h2_v0_j'+str(i)] = ('Local quantities/Auxiliary/Excitation/Level densities', 9+i, float)
        for i in range(3):
            self.pardict[f'pop_c_el3p_j' + str(i)] = ('Local quantities/Auxiliary/Excitation/Level densities', 427 + i, float)
        for i in range(8):
            self.pardict[f'pop_co_v0_j' + str(i)] = ('Local quantities/Auxiliary/Excitation/Level densities', 268 + i, float)
        for el, ind in zip(['h', 'h2', 'hd', 'co', 'oh'], [0, 2, 3, 46, 36]):
            self.pardict[f'cd_prof_{el}'] = ('Local quantities/Densities/Column densities', ind, float)
        for el, ind in zip(['h', 'h2', 'hd', 'co', 'oh'], [0, 2, 3, 46, 36]):
            self.pardict[f'cd_{el}'] = ('Integrated quantities/Column densities', ind, float)

    def read(self, show_meta=False, show_summary=True, fast=None):
        """
        Read model data from hdf5 file

        :param
            -  show_meta           :  if True, show Metadata table
            -  show_summary        :  if True, show summary

        :return: None
        """
        self.file = h5py.File(self.folder + self.filename, 'r')

        if show_meta:
            self.showMetadata()

        if fast is not None:
            self.fastread = fast

        # >>> model input parameters
        self.Z = self.par('metal')
        self.P = self.par('gas_pressure_input')
        self.n0 = self.par('proton_density_input')
        self.uv = self.par('radm_ini')
        self.cr = self.par('zeta')

        # >>> profile of physical quantities
        self.x = self.par('distance')

        for el in ['h', 'h2', 'hd', 'co']:
            setattr(self, el, self.par(f'cd_prof_{el}'))
        self.NCO = self.par('cd_co')
        self.av = self.par('av')
        self.Av = self.par('avmax')
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

        self.file.close()
        self.file = None

        if show_summary:
            self.show_summary()

    def par(self, par=None):
        """
        Read parameter or array from the hdf5 file, specified by Metadata name

        :param:
            -  par          :  the name of the parameter to read

        :return: x
            - x             :  corresponding data (string, number, array) correspond to the parameter
        """
        if self.file == None:
            self.file = h5py.File(self.folder + self.filename, 'r')

        meta = self.file['Metadata/Metadata']
        if par is not None:
            if hasattr(self, par):
                return getattr(self, par)
            elif self.fastread and par in self.pardict:
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
            if isinstance(x[0], float):
                return x[0]
            else:
                return typ(x[0])
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

    def show_summary(self, pars=['me', 'n', 'uv'], output=False):
        print('model: ' + self.name)
        if output:
            f = open(self.name+'dat', 'w')
        if 'all' in pars:
            self.file = h5py.File(self.folder + self.filename, 'r')
            meta = self.file['Metadata/Metadata']
            for ind in np.where(meta[:, 0] == b'/Parameters')[0]:
                attr = meta[ind, 0].decode() + '/' + meta[ind, 1].decode()
                typ = {'string': str, 'real': float, 'integer': int}[meta[ind, 5].decode()]
                #if len(x) == 1:
                print(meta[ind, 4].decode(), ' : ', typ(self.file[attr][:, int(meta[ind, 2].decode())][0].decode()))
                if output:
                    f.write(meta[ind, 4].decode() + ': ' + self.file[attr][:, int(meta[ind, 2].decode())][0].decode() + '\n')
            self.file.close()
        else:
            for p in pars:
                if hasattr(self, p):
                    print(p, ' : ', getattr(self, p))
                else:
                    print(p, ' : ', self.par(p))
                #print("{0:s} : {1:.2f}".format(p, getattr(self, p)))
        if output:
            f.close()

    def plot_model(self, parx='x', pars=['tgas', 'n', 'av'], species=None, logx=True, logy=True, fig=None, legend=True,
                   limit=None, relative_abund=None, ls='-', lw=2, indent=0.05):
        """
        Plot the model quantities

        :parameters:
            -  pars            :  list of the parameters to plot
            -  species         :  list of the lists of species to plot,
                                      e.g. [['H', 'H+', 'H2'], ['NH2j0', 'NH2j1', 'NH2j2']]
                                      each list will be plotted in independently
            -  logx            :  if True x in log10
            -  fig             :  fig object to plot
            -  legend          :  show Legend
            -  parx            :  what is on x axis
            -  limit           :  if not None plot only part of the cloud specified by limit
                                  e.g. {'NH2': 19.5}
            -  relative_abund  :  if True then plot relative abundances
            -  ls              :  line styles
            -  lw              :  line widths

        :return: fig
            -  fig            :  figure object
        """

        n = 1
        if species is not None:
            if sum([isinstance(s, list) for s in species]) == 0:
                species = [species]
            n += len(species)

        if fig is None:
            new_fig = True
            fig = plt.figure(figsize=(2 + 8*n, 10))
        else:
            new_fig = False

        l = 0
        if len(pars) > 0:
            if new_fig:
                ax = fig.add_axes([l + 0.04, 0.08, (1 - l) / (n + 1), 0.92])
                l = l + 1.3 * (1 - l) / (n + 1)
            else:
                ax = fig.axes[l]
                l += 2
            self.plot_phys_cond(pars=pars, logx=logx, ax=ax, legend=legend, parx=parx, limit=limit)

        if species is not None:
            if new_fig:
                l += indent
                width = (1 - l) / (n - np.sign(l))
            for i, sp in enumerate(species):
                if new_fig:
                    ax = fig.add_axes([l + indent, 0.08, width - indent, 0.92])
                    l += width
                else:
                    ax = fig.axes[l]
                    l += 1
                self.plot_profiles(species=sp, logx=logx, logy=logy, ax=ax, legend=legend, ylabel=False, parx=parx,
                                   limit=limit, relative_abund=relative_abund, ls=ls, lw=lw)

        fig.tight_layout()

        return fig

    def plot_phys_cond(self, pars=['tgas', 'n', 'av'], logx=True, ax=None, legend=True, parx='x', limit=None):
        """
        Plot the physical parameters in the model

        :parameters:
            -  pars          :  list of the parameters to plot
            -  logx          :  if True x in log10
            -  ax            :  axes object to plot
            -  legend        :  show Legend
            -  parx          :  what is on x axis
            -  limit         :  if not None plot only part of the cloud specified by limit
                                e.g. {'NH2': 19.5}

        :return: ax
            -  ax            :  axes object
        """

        if parx == 'av':
            xlabel = 'Av'
        elif parx == 'x':
            xlabel = 'log(Distance), cm'
        elif parx == 'h2':
            xlabel = 'log(NH2), cm-2'
        elif parx == 'hi':
            xlabel = 'log(NHI), cm-2'
        elif parx == 'co':
            xlabel = 'log(NCO), cm-2'

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        if logx:
            mask = getattr(self, parx) > 0
        else:
            mask = getattr(self, parx) > -1

        if limit is not None:
            if hasattr(self, list(limit.keys())[0]):
                v = getattr(self, list(limit.keys())[0])
                mask *= v < list(limit.values())[0]
            elif list(limit.keys())[0] in self.sp.keys():
                v = self.sp[list(limit.keys())[0]]
                mask *= v < list(limit.values())[0]
            else:
                warnings.warn('limit key {:s} is not found in the model. Mask did not applied'.format(list(limit.keys())[0]))

        if logx:
            x = np.log10(getattr(self, parx)[mask])
        else:
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

    def plot_profiles(self, species=None, parx='av', logx=False, logy=False, ax=None, label=None, ylabel=True,
                      legend=True, ls='-', lw=1, limit=None, relative_abund=None):
        """
        Plot the profiles of the species

        :param:
            -  species       :  list of the species to plot
            -  parx          :  what is on x axis
            -  logx          :  log of x axis
            -  logy          :  log of y axis
            -  ax            :  axis object to plot in, if None, then figure is created
            -  label         :  set label for the lines
            -  ylabel        :  set label of y axis
            -  legend        :  show legend
            -  ls            :  linestyles
            -  lw            :  linewidth
            -  limit         :  if not None plot only part of the cloud specified by limit
                                e.g. {'NH2': 19.5}

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
        elif parx == 'hi':
            xlabel = 'log(NHI), cm-2'
        elif parx == 'co':
            xlabel = 'log(NCO), cm-2'

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        if logx:
            mask = getattr(self, parx) > 0
        else:
            mask = getattr(self, parx) > -1

        if limit is not None:
            if hasattr(self, list(limit.keys())[0]):
                v = getattr(self, list(limit.keys())[0])
                mask *= v < list(limit.values())[0]
            elif list(limit.keys())[0] in self.sp.keys():
                v = self.sp[list(limit.keys())[0]]
                mask *= v < list(limit.values())[0]
            else:
                warnings.warn('limit key {:s} is not found in the model. Mask did not applied'.format(list(limit.keys())[0]))

        if logx:
            x = np.log10(getattr(self, parx)[mask])
        else:
            x = getattr(self, parx)[mask]

        ax.set_xlim([x[0], x[-1]])
        ax.set_xlabel(xlabel)


        if ax is None:
            fig, ax = plt.subplots()

        lab = label

        for s in species:
            print(s)
            #print(self.sp)
            if '/' in s:
                s1, s2 = s.split('/')[:]
                y = self.sp[s1][mask]
                d = self.sp[s2][mask]
            else:
                y = self.sp[s][mask]
                d = np.ones_like(y)

            if relative_abund is not None:
                d = getattr(self, relative_abund)

            if label is None:
                lab = s

            if logy:
                y = np.log10(y/d)

            p = ax.plot(x, y, ls=ls, label=lab, lw=lw)

            print(legend, isinstance(legend, int), type(legend))
            if type(legend) == int:
                ax.text(x[legend], y[legend], splatex[s],  va='bottom', ha='left', color=p[0].get_color())

        if ylabel:
            if relative_abund is None:
                ax.set_ylabel(r'number density, cm$^{-3}$', fontsize=20)
            else:
                ax.set_ylabel(r'$n_{i}/$'+relative_abund, fontsize=20)

        if legend is True:
            ax.legend()

        return ax

    def calc_cols(self, species=[], logN=np.inf, sides=2):
        """
        Calculate column densities for given species

        :param:
            -  species       :  list of the species to plot
            -  logN          :  column density threshold, dictionary with species and logN value
            -  side          :  make calculation to be one or two sided

        :return: sp
            -  sp            :  dictionary of the column densities by species
        """

        cols = OrderedDict()

        if logN == np.inf:
            for s in species:
                cols[s] = np.log10(integrate.cumtrapz(self.sp[s], x=self.x))
        else:
            if logN is not None:
                self.set_mask(species=list(logN.keys())[0], logN=logN[list(logN.keys())[0]] - np.log10((sides > 1) + 1), sides=sides)

            for s in species:
                cols[s] = np.log10(np.trapz(self.sp[s][self.mask], x=self.x[self.mask])) + np.log10((sides > 1) + 1)

        self.cols = cols

        return self.cols

    def set_mask(self, species='H', logN=None, sides=2):
        """
        Calculate mask for a given threshold

        :param:
            -  logN          :  column density threshold

        :return: None
        """
        cols = np.log10(self.sp['N' + species])
        #print(cols)

        if logN is not None and sides != 0:
            l = int(len(self.x) / sides) + 1 if sides > 1 else len(self.x)
            if logN > cols[l - 1]:
                logN = cols[l - 1]
            self.mask = cols < logN
        else:
            self.mask = cols > -1
        #return np.searchsorted(cols, value)
        #print('av_max:', self.av[self.mask][-1])

    def lnLike(self, species={}, syst=0, verbose=False, relative=None):
        lnL = 0
        if 0:
            #verbose:
            self.showSummary()
        for k, v in species.items():
            if relative is None:
                v1 = v
                s = self.cols[k]
            else:
                v1 = v / species[relative]
                s = self.cols[k] - self.cols[relative]

            if syst > 0:
                v1 *= a(0, syst, syst, 'l')
            if verbose:
                print(np.log10(self.uv), np.log10(self.n0), self.cols[k], v1.log(), v1.lnL(self.cols[k]))
            if v.type in ['m', 'u', 'l']:
                lnL += v1.lnL(s)

        self.lnL = lnL
        return self.lnL

class H2_exc():
    def __init__(self, folder='', H2database='all'):
        self.folder = folder if folder.endswith('/') else folder + '/'
        self.models = {}
        self.species = ['H', 'H+', 'H2', 'H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5', 'H2j6', 'H2j7', 'H2j8', 'H2j9', 'H2j10',
                        'NHI', 'NH2', 'NH2j0', 'NH2j1', 'NH2j2', 'NH2j3', 'NH2j4', 'NH2j5', 'NH2j6', 'NH2j7', 'NH2j8', 'NH2j9', 'NH2j10',
                        'HD', 'HDj0', 'HDj1',
                        'C', 'C+', 'CO', 'CIj0', 'CIj1', 'CIj2', 'NCO',
                        'COj0', 'COj1', 'COj2', 'COj3', 'COj4', 'COj5', 'COj6',
                        'SiII', 'SiIIj0', 'SiIIj1',
                        'O', 'O+', 'OH', 'NOH',
                        'H2_dest_rate', 'H2_form_rate_er', 'H2_form_rate_lh', 'H2_photo_dest_prob']
        self.readH2database(H2database)

    def readH2database(self, data='all'):
        import sys
        sys.path.append('/home/toksovogo/science/codes/python/3.5/')
        import H2_summary

        self.H2 = H2_summary.load_empty()
        if data == 'all':
            self.H2.append(H2_summary.load_QSO())
        if data in ['secret']:
            self.H2.append(H2_summary.load_secret())
        if data in ['CO']:
            self.H2.append(H2_summary.load_CO())

    def readmodel(self, filename=None, show_summary=False, folder=None):
        """
        Read one model by filename
        :param:
            -  filename             :  filename contains the model
            -  print_summary        :  if True, print summary for each model
        """
        if folder == None:
            folder = self.folder

        if filename is not None:
            m = model(folder=folder, filename=filename, species=self.species, show_summary=False)
            self.models[m.name] = m
            self.current = m.name

            if show_summary:
                m.showSummary()

    def readfolder(self, verbose=False):
        """
        Read list of models from the folder
        """
        if 1:
            for (dirpath, dirname, filenames) in os.walk(self.folder):
                print(dirpath, dirname, filenames)
                for f in filenames:
                    if f.endswith('.hdf5'):
                        self.readmodel(filename=f, folder=dirpath + '/')
        else:
            for f in os.listdir(self.folder):
                if f.endswith('.hdf5'):
                    self.readmodel(f, show_summary=verbose)


    def setgrid(self, pars=[], fixed={}, show=True):
        """
        Show and mask models for grid of specified parameters
        :param:
            -  pars           :  list of parameters in the grid, e.g. pars=['uv', 'n0']
            -  fixed          :  dict of parameters to be fixed, e.g. fixed={'z': 0.1}
            -  show           :  if true show the plot for the grid
        :return: mask
            -  mask           :  list of names of the models in the grid
        """
        self.grid = {p: [] for p in pars}
        self.mask = []
        self.grid['NH2tot'] = None

        for name, model in self.models.items():
            for k, v in fixed.items():
                if getattr(model, k) != v and v != 'all':
                    break
            else:
                for p in pars:
                    #if 'N' in p:
                    #    self.grid[p].append(np.log10(model.par(p)))
                    #else:
                    self.grid[p].append(model.par(p))
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
        Return component object from self.H2
        :param:
            -  object         :  object name.
                                    Examples: '0643' - will search for the 0643 im quasar names. Return first component.
                                              '0643_1' - will search for the 0643 im quasar names. Return second component
        :return: q
            -  q              :  qso.comp object (see file H2_summary.py how to retrieve data (e.g. column densities) from it
        """
        q = self.H2.get(object.split('_')[0])
        if q is not None:
            if len(object.split('_')) > 1:
                q = q.comp[int(object.split('_')[1])]
            else:
                q = q.comp[0]
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

    def compare(self, object='', species='', models='current', syst=0.0, syst_factor=1, levels=[], others='ignore', relative=False, sides=2):
        """
        Calculate the column densities of H2 rotational levels for the list of models given the total H2 column density.
        and also log of likelihood


        :param:
            -  object            :  object name
            -  species           :  which species to use. Can be 'H2' or 'CI'
            -  models            :  names of the models, can be list or string
            -  syst              :  add systematic uncertainty to the calculation of the likelihood
            -  syst_factor       :  multiply each uncertainty by this factor
            -  levels            :  levels that used to constraint, if empty list used all avaliable
            -  others            :  what to do with other levels (can be 'ignore', 'upper', 'lower')
            -  sides             :  how many sides are illuminated, can be 1 and 2 (for both sides)

        :return: None
            column densities are stored in the dictionary <cols> attribute for each model
            log of likelihood value is stored in <lnL> attribute
        """

        q = self.comp(object)
        if species == 'H2':
            if len(levels) > 0:
                full_keys = [s for s in q.e.keys() if ('H2j' in s) and ('v' not in s)]
                keys = ['H2j{:}'.format(i) for i in levels if 'H2j{:}'.format(i) in full_keys]
                spec = OrderedDict([(s, a(q.e[s].col.log().val, q.e[s].col.log().plus * syst_factor, q.e[s].col.log().minus * syst_factor, 'l') * a(0.0, syst, syst)) for s in keys])
                if others in ['lower', 'upper']:
                    for k in full_keys:
                        if k not in keys:
                            v = a(q.e[k].col.val, q.e[k].col.plus * syst_factor, q.e[k].col.minus * syst_factor)
                            print(q.e[k].col, v)
                            if syst > 0:
                                v = v * a(0.0, syst, syst)
                            if others == 'lower':
                                spec[k] = a(v.val - v.minus, t=others[0])
                            else:
                                spec[k] = a(v.val + v.plus, t=others[0])
            logN = {'H2': q.e['H2'].col.val}
            relative = 'H2j0' if relative else None

        elif species == 'CI':
            print('compare CI ', object)
            if len(levels) > 0:
                full_keys = [s for s in q.e.keys() if ('CIj' in s) and ('v' not in s)]
                keys = ['CIj{:}'.format(i) for i in list(set(levels) & set([0, 1, 2])) if 'CIj{:}'.format(i) in full_keys]
                print(keys)
            spec = OrderedDict([(s, q.e[s].col * a(0.0, syst, syst)) for s in full_keys])
            logN = {'H2': q.e['H2'].col.val}
            relative = 'CIj0' if relative else None

        elif species == 'CO':
            print('compare CO', object)
            if len(levels) > 0:
                full_keys = [s for s in q.e.keys() if ('COj' in s) and ('v' not in s)]
                keys = ['COj{:}'.format(i) for i in levels if 'COj{:}'.format(i) in full_keys]
                print(keys)
            logN = {'CO': q.e['CO'].col.val}
            relative = 'COj0' if relative else None
            #logN = {'CO': q.e['CO'].col.val}

        else:
            keys, full_keys, logN = [], [], None

        print(logN, sides, relative)

        spec = OrderedDict([(s, q.e[s].col * a(0.0, syst, syst)) for s in full_keys])

        for model in self.listofmodels(models):
            model.calc_cols(spec.keys(), logN=logN, sides=sides)
            #relative = 'COj0' if species == 'CO' else None
            #relative = None
            #model.lnLike(OrderedDict([(s, q.e[s].col * a(0.0, syst, syst)) for s in keys]), relative=relative)
            model.lnLike(spec, relative=relative)

    def comparegrid(self, object='', species='', pars=[], fixed={}, syst=0.0, syst_factor=1.0, plot=True, show_best=True, levels='all', others='ignore', relative=False, sides=2):
        print('comparegrid', syst, syst_factor, pars, fixed)
        self.setgrid(pars=pars, fixed=fixed, show=False)
        if object != '':
            self.grid['NH2tot'] = self.comp(object).e['H2'].col.val
        #print(others)
        self.compare(object, species=species, models=self.mask, syst=syst, syst_factor=syst_factor, levels=levels, others=others, relative=relative, sides=sides)
        self.grid['lnL'] = np.asarray([self.models[m].lnL for m in self.mask])
        self.grid['cols'] = np.asarray([self.models[m].cols for m in self.mask])
        self.grid['dims'] = len(pars)
        #print(self.grid)

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

    def plot_models(self, ax=None, models='current', logN=None, species='<7', legend=False, sides=2):
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
                m.calc_cols(species, logN={'H2': logN}, sides=sides)

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

    def plot_one_parameter_vary(self, parx='x', pary='NH2', ax=None, pars=None, fixed={}):
        if ax is None:
            fig, ax = plt.subplots()

        species = [pary]

        models = self.setgrid(pars=[pars], fixed=fixed)
        for m in self.listofmodels(models):
            print(m.name)
            m.plot_profiles(species=species, logx=True, logy=True, label=m.name, ax=ax, legend=False, parx=parx, limit=None)

        return ax

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

    if 1:
        folder = 'data_av_full/'
        filename = 'pdr_grid_av01_n1e1_uv3e0_s_20_s_20.hdf5'  # 'AGN_n4_00_UV3_00_s_10.hdf5'
        m = model(folder=folder, filename=filename, show_meta=True,
                  species=['H', 'H+', 'H2', 'H2j0', 'H2j1', 'HD', 'HDj0', 'HDj1', 'D', 'CIj0', 'CIj1', 'CIj2', 'SiI',
                           'SiII', 'SiIII', 'SiIIj0', 'SiIIj1', 'C+', 'NH2', 'NHD', 'NSiII', 'H2_diss', 'Av', 'NCOtot'])

    if 0:
        folder = 'C:/science/Noterdaeme/Coronographic/J0015+1842/JWST/' #'data_J0015/exact/'
        filename = 'J0015_UV3_n4_Av0_5_s_20.hdf5'  #'AGN_n4_00_UV3_00_s_10.hdf5'
        m = model(folder=folder, filename=filename, show_meta=False,
                  species=['H', 'H+', 'H2', 'H2j0', 'H2j1', 'HD', 'HDj0', 'HDj1', 'D', 'CIj0', 'CIj1', 'CIj2', 'SiI', 'SiII', 'SiIII', 'SiIIj0', 'SiIIj1', 'C+', 'NH2', 'NHD', 'NSiII', 'H2_diss', 'Av'])

        #m.plot_model(parx='x', pars=['tgas', 'n'], species=[['H', 'H+', 'H2', 'Av'], ['SiI', 'SiII', 'SiIII', 'SiIIj0', 'SiIIj1']], logx=True, logy=True)

        fig, ax = plt.subplots()
        if 1:
            print(m.par('avmax'))
            print('S5', m.par('inta_00_h2_v0_j7__v0_j5'))
            print('1S4', m.par('inta_00_h2_v1_j4__v0_j2'))
            print('1S3', m.par('inta_00_h2_v1_j3__v0_j1'))
            print('1S2', m.par('inta_00_h2_v1_j2__v0_j0'))
            print('1Q4', m.par('inta_00_h2_v1_j4__v0_j4'))
            print('1Q3', m.par('inta_00_h2_v1_j3__v0_j3'))
            print('1Q2', m.par('inta_00_h2_v1_j2__v0_j2'))
            print('1Q1', m.par('inta_00_h2_v1_j1__v0_j1'))
            print('1O0', m.par('inta_00_h2_v1_j0__v0_j2'))
            print('1O1', m.par('inta_00_h2_v1_j1__v0_j3'))
            print('1S2', m.par('inta_00_h2_v1_j4__v0_j2'))
            print('1S3', m.par('inta_00_h2_v1_j5__v0_j3'))
        if 0:
            l = m.par('wavelength')
            init = m.par('spec_int_tot')
            uv = m.par('uv_flux')
            print(len(l), len(uv))
            ax.plot(np.log10(l), np.log10(init))
        if 0:
            m.calc_cols(species=['SiIIj0', 'SiIIj1'])
            #ax.plot(np.log10(m.x[1:]), m.cols['SiIIj0'] - m.cols['SiIIj0'], '-k')
            ax.plot(np.log10(m.x[1:]), m.cols['SiIIj0'], '-k')
            ax.plot(np.log10(m.x[1:]), m.cols['SiIIj1'], '-r')
        if 0:
            ax.plot(np.log10(m.sp['NSiII']), np.log10(m.sp['SiIIj1'] / m.sp['SiIIj0']))
            ax.set_xlabel('logN(SiII)')
            ax.set_ylabel('log N(SiIIj1) / N(SiIIj0)')

        #m.plot_phys_cond(pars=['tgas', 'n', 'N_H2'])
        #m.plot_profiles(species=['H2', 'HD', 'Cj0', 'Cj1', 'Cj2'], logx=True, logy=True, parx='x')
        #print(m.x, m.h2, m.hd)

    if 0:
        fig, ax = plt.subplots()
        H2 = H2_exc(folder='C:/Users/Serj/Desktop/Meudon/')
        if 1:
            H2.readfolder()
            if 1:
                H2.plot_models(ax=ax, models='all')
                H2.compare_models(speciesname=['NH2j0'], models='all', physcondname=['tgas'], logy=True,
                                  parx='av')  # physcondname=['tgas','n'],
                #H2.compare_models(speciesname=['NH2'], models='all', logy=True, parx='av')
                # H2.compare_models(speciesname=['H2_photo_dest_prob'], models='all', logy=True, parx='x')
                # H2.compare_models(speciesname=['NH2'], models='all', logy=True)
                # H2.compare_models(speciesname=['NH2'], models='all', logy=True, parx='x')
                # H2.compare_models(speciesname=['H2'], models='all', logy=True, parx='x')
            else:
                H2.plot_models(ax=ax, models=name)

        H2.plot_objects(objects=H2.H2.all())

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



