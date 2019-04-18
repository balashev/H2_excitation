from functools import partial
from matplotlib import cm
import matplotlib.pyplot as plt
import pickle
from PyQt5.QtCore import (Qt, )
from PyQt5.QtGui import (QFont, )
from PyQt5.QtWidgets import (QApplication, QMessageBox, QMainWindow, QSplitter, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QPushButton, QHeaderView, QCheckBox,
                             QRadioButton, QButtonGroup, QComboBox, QTableView, QLineEdit)
import pyqtgraph as pg
from scipy.interpolate import interp2d, RectBivariateSpline, Rbf
import sys
sys.path.append('C:/science/python')
from H2_exc import *
from spectro.stats import distr2d
from spectro.a_unc import a
import copy

class image():
    """
    class for working with images (2d spectra) inside Spectrum plotting
    """
    def __init__(self, x=None, y=None, z=None, err=None, mask=None):
        if any([v is not None for v in [x, y, z, err, mask]]):
            self.set_data(x=x, y=y, z=z, err=err, mask=mask)
        else:
            self.z = None

    def set_data(self, x=None, y=None, z=None, err=None, mask=None):
        for attr, val in zip(['z', 'err', 'mask'], [z, err, mask]):
            if val is not None:
                setattr(self, attr, np.asarray(val))
            else:
                setattr(self, attr, val)
        if x is not None:
            self.x = np.asarray(x)
        else:
            self.x = np.arange(z.shape[0])
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = np.arange(z.shape[1])

        self.pos = [self.x[0] - (self.x[1] - self.x[0]) / 2, self.y[0] - (self.y[1] - self.y[0]) / 2]
        self.scale = [(self.x[-1] - self.x[0]) / (self.x.shape[0]-1), (self.y[-1] - self.y[0]) / (self.y.shape[0]-1)]
        for attr in ['z', 'err']:
            self.getQuantile(attr=attr)
            self.setLevels(attr=attr)


    def getQuantile(self, quantile=0.997, attr='z'):
        if getattr(self, attr) is not None:
            x = np.sort(getattr(self, attr).flatten())
            x = x[~np.isnan(x)]
            setattr(self, attr+'_quantile', [x[int(len(x)*(1-quantile)/2)], x[int(len(x)*(1+quantile)/2)]])
        else:
            setattr(self, attr + '_quantile', [0, 1])

    def setLevels(self, bottom=None, top=None, attr='z'):
        quantile = getattr(self, attr+'_quantile')
        if bottom is None:
            bottom = quantile[0]
        if top is None:
            top = quantile[1]
        top, bottom = np.max([top, bottom]), np.min([top, bottom])
        if top - bottom < (quantile[1] - quantile[0]) / 100:
            top += ((quantile[1] - quantile[0]) / 100 - (top - bottom)) /2
            bottom -= ((quantile[1] - quantile[0]) / 100 - (top - bottom)) / 2
        setattr(self, attr+'_levels', [bottom, top])

    def find_nearest(self, x, y, attr='z'):
        z = getattr(self, attr)
        if len(z.shape) == 2:
            return z[np.min([z.shape[0]-1, (np.abs(self.y - y)).argmin()]), np.min([z.shape[1]-1, (np.abs(self.x - x)).argmin()])]
        else:
            return None


class plotExc(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'log(N/g)', 'bottom': 'Energy, cm-1'})
        self.initstatus()
        self.vb = self.getViewBox()
        self.view = {}
        self.models = {}
        self.legend = pg.LegendItem(offset=(-70, 30))
        self.legend.setParentItem(self.vb)
        self.legend_model = pg.LegendItem(offset=(-70, -30))
        self.legend_model.setParentItem(self.vb)

    def initstatus(self):
        pass

    def add(self, name, add, species=[]):
        if add:
            q = self.parent.H2.comp(name)
            if species is None or len(species) == 0:
                sp = [s for s in q.e.keys() if 'H2j' in s]
            j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
            x = [H2energy[0, i] for i in j]
            y = [q.e['H2j' + str(i)].col / stat[i] for i in j]
            typ = [q.e['H2j' + str(i)].col.type for i in j]
            self.view[name] = [pg.ErrorBarItem(x=np.asarray(x), y=column(y, 'v'), top=column(y, 'p'), bottom=column(y, 'm'), beam=2),
                               pg.ScatterPlotItem(x, column(y, 'v'), symbol='o', size=15)]
            self.vb.addItem(self.view[name][0])
            self.vb.addItem(self.view[name][1])
            self.legend.addItem(self.view[name][1], name)
            self.redraw()
        else:
            try:
                self.vb.removeItem(self.view[name][0])
                self.vb.removeItem(self.view[name][1])
                self.legend.removeItem(self.view[name][1])
                del self.view[name]
            except:
                pass

    def add_model(self, name, add=True):
        if add:
            for ind, m in enumerate(self.parent.H2.listofmodels(name)):
                j = np.sort([int(s[3:]) for s in m.cols.keys()])
                x = [H2energy[0, i] for i in j]
                mod = [m.cols['H2j'+str(i)] - np.log10(stat[i]) for i in j]
            self.models[name] = pg.PlotCurveItem(x, mod)
            self.vb.addItem(self.models[name])
            self.legend_model.addItem(self.models[name], name)
            self.redraw()
        else:
            try:
                self.vb.removeItem(self.models[name])
                self.legend_model.removeItem(self.models[name])
                del self.models[name]
            except:
                pass

    def add_temp(self, cols=None, pars=None, add=True):
        if add:
            j = np.sort([int(s[3:]) for s in cols.keys()])
            x = [H2energy[0, i] for i in j]
            mod = [cols['H2j'+str(i)] - np.log10(stat[i]) for i in j]
            if 1:
                print(cols['H2j'+str(i)] for i in j)
                grid = self.parent.H2.grid
                species = {}
                sp = grid['cols'][0].keys()
                for s in sp:
                    v1 = self.parent.H2.comp(grid['name']).e[s].col.log().copy()
                    # if kind == 'fast':
                    #    v1.minus, v1.plus = np.sqrt(v1.minus ** 2 + float(self.addSyst.text()) ** 2), np.sqrt(
                    #        v1.plus ** 2 + float(self.addSyst.text()) ** 2)
                    # elif kind == 'accurate':
                    v1 *= a(0, 0.2, 0.2, 'l')
                    #v1.plus, v1.minus = 0.2, 0.2
                    species[s] = v1
                lnL = 0
                for s, v in species.items():
                    if v.type == 'm':
                        lnL += v.lnL(cols[s])
                print(lnL)

            self.temp_model = pg.PlotCurveItem(x, mod)
            self.vb.addItem(self.temp_model)
            text = 'selected'
            if pars is not None:
                text += ' {0:.2f} {1:.2f}'.format(pars[0], pars[1])
            self.legend_model.addItem(self.temp_model, text)
        else:
            try:
                self.vb.removeItem(self.temp_model)
                self.legend_model.removeItem(self.temp_model)
            except:
                pass


    def redraw(self):
        for i, v in enumerate(self.view.values()):
            v[1].setBrush(pg.mkBrush(cm.rainbow(0.01 + 0.98 * i / len(self.view), bytes=True)[:3] + (255,)))
        for i, v in enumerate(self.models.values()):
            v.setPen(pg.mkPen(cm.rainbow(0.01 + 0.98 * i / len(self.models), bytes=True)[:3] + (255,)))

class textLabel(pg.TextItem):
    def __init__(self, parent, text, x=None, y=None, name=None):
        self.parent = parent
        self.text = text
        self.active = False
        self.name = name
        pg.TextItem.__init__(self, text=text, fill=pg.mkBrush(0, 0, 0, 0), anchor=(0.5,0.5))
        self.setFont(QFont("SansSerif", 16))
        self.setPos(x, y)
        self.redraw()

    def redraw(self):
        if self.active:
            self.setColor((255, 225, 53))
        else:
            self.setColor((255, 255, 255))
        self.parent.parent.plot_exc.add_model(self.name, add=self.active)

    def plot_model(self):
        print(self.parent.parent.H2.grid['NH2tot'])
        m = self.parent.parent.H2.listofmodels(self.name)[0]
        m.plot_model(parx='x', pars=['tgas', 'n'],
                     species=[['H', 'H+', 'H2', 'H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5', 'Cj0', 'Cj1', 'Cj2'],
                              ['NH2j0', 'NH2j1', 'NH2j2', 'NH2j3', 'NH2j4', 'NH2j5']],
                     logx=True, logy=True, limit={'NH2': 10**self.parent.parent.H2.grid['NH2tot'] / 2 })
        plt.show()

    def mouseClickEvent(self, ev):

        if (QApplication.keyboardModifiers() == Qt.ShiftModifier):
            self.active = not self.active
            self.redraw()

        if (QApplication.keyboardModifiers() == Qt.ControlModifier):
            self.plot_model()

    def clicked(self, pts):
        print("clicked: %s" % pts)

class plotGrid(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'log(n)', 'bottom': 'log(UV)'})
        self.initstatus()
        self.vb = self.getViewBox()
        self.image = None
        self.text = None
        self.grid = image()
        cdict = cm.get_cmap('viridis')
        cmap = np.array(cdict.colors)
        cmap[-1] = [1, 0.4, 0]
        map = pg.ColorMap(np.linspace(0, 1, cdict.N), cmap, mode='rgb')
        self.colormap = map.getLookupTable(0.0, 1.0, 256, alpha=False)

    def initstatus(self):
        self.s_status = True
        self.selected_point = None

    def set_data(self, x=None, y=None, z=None, view='text'):
        if view == 'text':
            if self.text is not None:
                for t in self.text:
                    self.vb.removeItem(t)
            self.text = []
            for xi, yi, zi, name in zip(x, y, z, [self.parent.H2.models[m].name for m in self.parent.H2.mask]):
                self.text.append(textLabel(self, '{:.1f}'.format(zi), x=np.log10(xi), y=np.log10(yi), name=name))
                #self.text.append(pg.TextItem(html='<div style="text-align: center"><span style="color: #FF0; font-size: 16pt;">' + '{:.1f}'.format(zi) + '</span></div>'))
                #self.text[-1].setPos(np.log10(xi), np.log10(yi))
                self.vb.addItem(self.text[-1])

        if view == 'image':
            if self.image is not None:
                self.vb.removeItem(self.image)
            self.image = pg.ImageItem()
            self.grid.set_data(x=x, y=y, z=z)
            self.image.translate(self.grid.pos[0], self.grid.pos[1])
            self.image.scale(self.grid.scale[0], self.grid.scale[1])
            self.image.setLookupTable(self.colormap)
            self.image.setLevels(self.grid.levels)
            self.vb.addItem(self.image)

    def mousePressEvent(self, event):
        super(plotGrid, self).mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            if self.s_status:
                name = self.parent.H2.grid['name']
                self.mousePoint = self.vb.mapSceneToView(event.pos())
                print(self.mousePoint.x(), self.mousePoint.y())
                if self.parent.grid_pars.cols is not None:
                    cols = {}
                    sp = self.parent.H2.grid['cols'][0].keys()
                    lnL = 0
                    for s in sp:
                        v = self.parent.H2.comp(name).e[s].col
                        cols[s] = self.parent.grid_pars.cols[s](self.mousePoint.x(), self.mousePoint.y())
                        print(self.mousePoint.x(), self.mousePoint.y(),s, cols[s])
                        v1 = v * a(0, 0.2, 0.2, 'l')
                        if v.type == 'm':
                            lnL += v1.lnL(cols[s])
                    self.parent.plot_exc.add_temp(cols, add=False)
                    self.parent.plot_exc.add_temp(cols, pars=[self.mousePoint.x(), self.mousePoint.y()])

    def keyPressEvent(self, event):
        super(plotGrid, self).keyPressEvent(event)
        key = event.key()

        if not event.isAutoRepeat():
            if event.key() == Qt.Key_S:
                self.s_status = True

    def keyReleaseEvent(self, event):
        super(plotGrid, self).keyReleaseEvent(event)
        key = event.key()

        if not event.isAutoRepeat():

            if event.key() == Qt.Key_S:
                self.s_status = True
                self.parent.plot_exc.add_temp([], add=False)

class QSOlistTable(pg.TableWidget):
    def __init__(self, parent):
        super().__init__(editable=False, sortable=False)
        self.setStyleSheet(open('styles.ini').read())
        self.parent = parent
        self.format = None

        self.contextMenu.addSeparator()
        self.contextMenu.addAction('results').triggered.connect(self.show_results)

        self.resize(100, 1200)
        self.show()

    def setdata(self, data):
        self.data = data
        self.setData(data)
        if self.format is not None:
            for k, v in self.format.items():
                self.setFormat(v, self.columnIndex(k))
        self.resizeColumnsToContents()
        self.horizontalHeader().setResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setResizeMode(0, QHeaderView.ResizeToContents)
        self.horizontalHeader().setResizeMode(1, QHeaderView.ResizeToContents)
        w = 180 + self.verticalHeader().width() + self.autoScrollMargin()*1.5
        w += sum([self.columnWidth(c) for c in range(self.columnCount())])
        self.resize(w, self.size().height())
        self.setSortingEnabled(False)

    def compare(self):
        grid = self.parent.parent.grid_pars.pars
        syst = float(self.parent.parent.grid_pars.addSyst.text()) if self.parent.parent.grid_pars.addSyst.text().strip() != '' else 0
        pars = [list(grid.keys())[list(grid.values()).index('x')],
                list(grid.keys())[list(grid.values()).index('y')]]
        fixed = list(grid.keys())[list(grid.values()).index('fixed')]
        if getattr(self.parent.parent.grid_pars, fixed + '_val').currentText() != '':
            fixed = {fixed: float(getattr(self.parent.parent.grid_pars, fixed + '_val').currentText())}
        else:
            fixed = {fixed: 'all'}
        for idx in self.selectedIndexes():
            if idx.column() == 0:
                name = self.cell_value('name')
                self.parent.parent.H2.comparegrid(name, pars=pars, fixed=fixed, syst=syst, plot=False, levels=self.parent.parent.grid_pars.H2levels)
                grid = self.parent.parent.H2.grid
                self.parent.parent.H2.grid['name'] = name
                print('grid', grid['uv'], grid['n0'], grid['lnL'])
                self.parent.parent.plot_reg.set_data(x=grid[pars[0]], y=grid[pars[1]], z=grid['lnL'])
                self.parent.parent.plot_reg.setLabels(bottom='log('+pars[0]+')', left='log('+pars[1]+')')
                #self.pos = [self.x[0] - (self.x[1] - self.x[0]) / 2, self.y[0] - (self.y[1] - self.y[0]) / 2]
                #self.scale = [(self.x[-1] - self.x[0]) / (self.x.shape[0] - 1), (self.y[-1] - self.y[0]) / (self.y.shape[0] - 1)]

    def show_results(self):
        for idx in self.selectedIndexes():
            if idx.column() == 0:
                name = self.cell_value('name')
                str = "".join(['result/all/', name, '.pkl'])
                strnodes = "".join(['result/all/nodes/', name, '.pkl'])
                with open(strnodes, 'rb') as f:
                    x1, y1, z1 = pickle.load(f)
                with open(str, 'rb') as f:
                    x, y, z = pickle.load(f)
                    X, Y = np.meshgrid(x, y)
                    plt.subplot(1, 2, 1)
                    plt.pcolor(X, Y, z, cmap=cm.jet, vmin=-50, vmax=0)
                    plt.scatter(x1, y1, 100, z1, cmap=cm.jet, vmin=-50, vmax=0)  # ,edgecolors='black')
                    plt.title('RBF interpolation - likelihood')
                    plt.xlim(-1.2, 3.2)
                    plt.ylim(0.8, 5.2)
                    plt.colorbar()

                    d = distr2d(x=x, y=y, z=np.exp(z))
                    dx, dy = d.marginalize('y'), d.marginalize('x')
                    dx.stats(latex=2, name='log UV')
                    dy.stats(latex=2, name='log n')
                    # d.plot(color=None)
                    ax = plt.subplot(1, 2, 2)
                    d.plot_contour(ax=ax, color='lime', xlabel='$\log n$ [cm$^{-3}$]', ylabel='$\log I_{UV}$ [Draine field]')
                    d.plot(color='lime', xlabel='$\log n$ [cm$^{-3}$]', ylabel='$\log I_{UV}$ [Draine field]')

                    plt.show()

    def columnIndex(self, columnname):
        return [self.horizontalHeaderItem(x).text() for x in range(self.columnCount())].index(columnname)

    def cell_value(self, columnname, row=None):
        if row is None:
            row = self.currentItem().row()

        cell = self.item(row, self.columnIndex(columnname)).text()  # get cell at row, col

        return cell

class chooseH2SystemWidget(QWidget):
    """
    Widget for choose fitting parameters during the fit.
    """
    def __init__(self, parent, closebutton=True):
        super().__init__()
        self.parent = parent
        #self.resize(700, 900)
        #self.move(400, 100)
        self.setStyleSheet(open('styles.ini').read())

        self.saved = []

        layout = QVBoxLayout()

        self.table = QSOlistTable(self)
        data = self.parent.H2.H2.makelist(pars=['z_dla', 'Me__val', 'H2__val'], sys=self.parent.H2.H2.all(), view='numpy')
        #data = self.H2.H2.list(pars=['name', 'H2', 'metallicity'])
        self.table.setdata(data)
        self.table.setSelectionBehavior(QTableView.SelectRows);
        self.buttons = {}
        for i, d in enumerate(data):
            wdg = QWidget()
            l = QVBoxLayout()
            l.addSpacing(3)
            button = QPushButton(d[0], self, checkable=True)
            button.setFixedSize(100, 30)
            button.setChecked(False)
            button.clicked[bool].connect(partial(self.click, d[0]))
            self.buttons[d[0]] = button
            l.addWidget(button)
            l.addSpacing(3)
            l.setContentsMargins(0, 0, 0, 0)
            wdg.setLayout(l)
            self.table.setCellWidget(i, 0, wdg)

        layout.addWidget(self.table)

        self.scroll = None

        self.layout = QVBoxLayout()
        layout.addLayout(self.layout)

        if closebutton:
            self.okButton = QPushButton("Close")
            self.okButton.setFixedSize(110, 30)
            self.okButton.clicked[bool].connect(self.ok)
            hbox = QHBoxLayout()
            hbox.addWidget(self.okButton)
            hbox.addStretch()
            layout.addLayout(hbox)

        self.setLayout(layout)

    def click(self, name):
        self.parent.plot_exc.add(name, self.buttons[name].isChecked())
        self.table.setCurrentCell(np.where(self.table.data['name'] == name)[0][0], 0)

    def ok(self):
        self.hide()
        self.parent.chooseFit = None
        self.deleteLater()

    def cancel(self):
        for par in self.parent.fit.list():
            par.fit = self.saved[str(par)]
        self.close()


class gridParsWidget(QWidget):
    """
    Widget for choose fitting parameters during the fit.
    """
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        #self.resize(700, 900)
        #self.move(400, 100)
        self.pars = {'n0': 'x', 'uv': 'y', 'me': 'fixed'}
        self.parent.H2.setgrid(pars=list(self.pars.keys()), show=False)
        self.cols, self.x_, self.y_, self.z_ = None, None, None, None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('grid parameters:'))

        for n in self.pars.keys():
            l = QHBoxLayout(self)
            l.addWidget(QLabel((n + ': ')[:3]))
            self.group = QButtonGroup(self)
            for b in ('x', 'y', 'z', 'fixed'):
                setattr(self, b, QCheckBox(b, checkable=True))
                getattr(self, b).clicked[bool].connect(partial(self.setGridView, par=n, b=b))
                l.addWidget(getattr(self, b))
                self.group.addButton(getattr(self, b))
            getattr(self, self.pars[n]).setChecked(True)
            setattr(self, n + '_val', QComboBox(self))
            getattr(self, n + '_val').setFixedSize(80, 25)
            getattr(self, n + '_val').addItems(np.append([''], np.asarray(np.sort(np.unique(self.parent.H2.grid[n])), dtype=str)))
            if self.pars[n] is 'fixed':
                getattr(self, n + '_val').setCurrentIndex(1)
            l.addWidget(getattr(self, n + '_val'))
            l.addStretch(1)
            layout.addLayout(l)

        l = QHBoxLayout(self)
        l.addWidget(QLabel('add systematic unc.:'))
        self.addSyst = QLineEdit()
        self.addSyst.setText(str(0.2))
        self.addSyst.setFixedSize(90, 30)
        l.addWidget(self.addSyst)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout(self)
        self.compare = QPushButton('Compare')
        self.compare.clicked[bool].connect(self.compareIt)
        self.compare.setFixedSize(90, 30)
        l.addWidget(self.compare)
        self.H2levels = np.arange(6)
        self.levels = QLineEdit(" ".join([str(i) for i in self.H2levels]))
        self.levels.setFixedSize(90, 30)
        self.levels.editingFinished.connect(self.setLevels)
        l.addWidget(self.levels)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout(self)
        self.refine = QPushButton('Refine:')
        self.refine.clicked[bool].connect(partial(self.regridIt, kind='accurate'))
        self.refine.setFixedSize(90, 30)
        l.addWidget(self.refine)
        self.numPlot = QLineEdit(str(30))
        self.numPlot.setFixedSize(90, 30)
        l.addWidget(self.numPlot)
        l.addWidget(QLabel('x:'))
        self.xmin = QLineEdit('')
        self.xmin.setFixedSize(30, 30)
        l.addWidget(self.xmin)
        l.addWidget(QLabel('..'))
        self.xmax = QLineEdit('')
        self.xmax.setFixedSize(30, 30)
        l.addWidget(self.xmax)
        l.addWidget(QLabel('y:'))
        self.ymin = QLineEdit('')
        self.ymin.setFixedSize(30, 30)
        l.addWidget(self.ymin)
        l.addWidget(QLabel('..'))
        self.ymax = QLineEdit('')
        self.ymax.setFixedSize(30, 30)
        l.addWidget(self.ymax)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout(self)
        self.plot = QPushButton('Plot')
        self.plot.clicked[bool].connect(self.plotIt)
        self.plot.setFixedSize(90, 30)
        l.addWidget(self.plot)
        self.export = QPushButton('Export')
        self.export.clicked[bool].connect(self.exportIt)
        self.export.setFixedSize(90, 30)
        l.addWidget(self.export)
        l.addStretch(1)
        layout.addLayout(l)

        layout.addStretch(1)

        self.setLayout(layout)

        self.setStyleSheet(open('styles.ini').read())

    def setLevels(self):
        try:
            self.H2levels = [int(s) for s in self.levels.text().split()]
        except:
            pass

    def compareIt(self):
        self.parent.H2_systems.table.compare()
        self.interpolateIt()
        grid = self.parent.H2.grid
        x, y = np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('x')]]), np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('y')]])
        self.xmin.setText(str(np.min(x)))
        self.xmax.setText(str(np.max(x)))
        self.ymin.setText(str(np.min(y)))
        self.ymax.setText(str(np.max(y)))

    def interpolateIt(self):
        grid = self.parent.H2.grid
        x, y = np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('x')]]), np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('y')]])
        sp = grid['cols'][0].keys()
        self.cols = {}
        for s in sp:
            if 0:
                self.cols[s] = interp2d(x, y, [c[s] for c in grid['cols']], kind='cubic')
            if 0:
                xt, yt = np.unique(sorted(x)), np.unique(sorted(y))
                z = np.zeros([xt.shape[0], yt.shape[0]])
                for i, xi in enumerate(xt):
                    for k, yk in enumerate(yt):
                        z[i, k] = grid['cols'][np.argmin((xi - x) ** 2 + (yk - y) ** 2)][s]

                self.cols[s] = RectBivariateSpline(xt, yt, z, kx=2, ky=2)
            if 0:
                if s=='H2j0' or s=='H2j1':
                    self.cols[s] = Rbf(x, y, np.asarray([c[s] for c in grid['cols']]), function='inverse', epsilon=0.3)
                else:
                    self.cols[s] = Rbf(x, y, np.asarray([c[s] for c in grid['cols']]), function='multiquadric',smooth=0.2)
            if 1:
                self.cols[s] = Rbf(x, y, np.asarray([c[s] for c in grid['cols']]), function='multiquadric', smooth=0.1)


                #rbf = Rbf(x,y,z,function='multiquadric',smooth=0.2)


    def regridIt(self, kind='accurate', save=True):
        grid = self.parent.H2.grid
        num = int(self.numPlot.text())
        x, y = np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('x')]]), np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('y')]])
        x1, y1 = x, y # copy for save
        x, y = np.linspace(float(self.xmin.text()), float(self.xmax.text()), num), np.linspace(float(self.ymin.text()), float(self.ymax.text()), num)
        X, Y = np.meshgrid(x, y)
        z = np.zeros_like(X)
        sp = grid['cols'][0].keys()
        sp = [s for s in sp if int(s[3:]) in self.H2levels]
        species = {}
        for s in sp:
            v1 = self.parent.H2.comp(grid['name']).e[s].col.log().copy()
            if kind == 'fast':
                v1.minus, v1.plus = np.sqrt(v1.minus ** 2 + float(self.addSyst.text()) ** 2), np.sqrt(v1.plus ** 2 + float(self.addSyst.text()) ** 2)
            elif kind == 'accurate':
                v1 *= a(0, float(self.addSyst.text()), float(self.addSyst.text()), 'l')
            species[s] = v1
        if save:
            cols = {}
            for s in self.cols.keys():
                cols[s] = np.zeros_like(z)
        for i, xi in enumerate(x):
            for k, yi in enumerate(y):
                lnL = 0
                for s, v in species.items():
                    if v.type == 'm':
                        lnL += v.lnL(self.cols[s](xi, yi))
                        cols[s][k, i] = self.cols[s](xi, yi)
                z[k, i] = lnL
        self.x_, self.y_, self.z_ = x, y, z

        if save:
            for s in cols.keys():
                with open('temp/{:s}'.format(s), 'wb') as f:
                    pickle.dump([self.x_, self.y_, cols[s]], f)

                z1 = np.asarray([c[s] for c in grid['cols']])
                with open('temp/{:s}_nodes'.format(s), 'wb') as f:
                    pickle.dump([x1, y1, z1], f)

            lnL1 = np.asarray(grid['lnL'])
            with open('output/nodes/{:s}.pkl'.format(self.parent.H2.grid['name']), 'wb') as f:
                pickle.dump([x1, y1, lnL1], f)
            with open('temp/lnL_nodes.pkl', 'wb') as f:
                pickle.dump([x1, y1, lnL1], f)

#            with open('temp/{:s}lnL_nodes.pkl'.format(s), 'wb') as f:
#                pickle.dump([x1, y1, lnL1], f)

    def plotIt(self):
        if self.x is not None:
            d = distr2d(x=self.x_, y=self.y_, z=np.exp(self.z_))
            dx, dy = d.marginalize('y'), d.marginalize('x')
            dx.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('x')])
            dy.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('y')])
            d.plot(color=None)
            plt.show()


    def exportIt(self):
        with open('output/{:s}.pkl'.format(self.parent.H2.grid['name']), 'wb') as f:
            pickle.dump([self.x_, self.y_, self.z_], f)
        with open('temp/lnL.pkl', 'wb') as f:
            pickle.dump([self.x_, self.y_, self.z_], f)

    def setGridView(self, par, b):
        self.pars[par] = b
        if b is 'fixed':
            print(getattr(self, par + '_val').text())
            self.pars[par] = getattr(self, par + '_val').currentText()
        print(self.pars)

class H2viewer(QMainWindow):

    def __init__(self):
        super().__init__()
        #self.H2 = H2_exc(folder='data_z0.3')
        self.H2 = H2_exc(folder='data_01_temp', H2database='all')
        self.H2.readfolder()
        self.initStyles()
        self.initUI()

    def initStyles(self):
        self.setStyleSheet(open('styles.ini').read())

    def initUI(self):
        dbg = pg.dbg()
        # self.specview sets the type of plot representation

        # >>> create panel for plotting spectra
        self.plot_exc = plotExc(self)
        self.plot_reg = plotGrid(self)
        self.H2_systems = chooseH2SystemWidget(self, closebutton=False)
        self.grid_pars = gridParsWidget(self)
        # self.plot.setFrameShape(QFrame.StyledPanel)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter_plot = QSplitter(Qt.Vertical)
        self.splitter_plot.addWidget(self.plot_exc)
        self.splitter_plot.addWidget(self.plot_reg)
        self.splitter.addWidget(self.splitter_plot)
        self.splitter_pars = QSplitter(Qt.Vertical)
        self.splitter_pars.addWidget(self.H2_systems)
        self.splitter_pars.addWidget(self.grid_pars)
        self.splitter_pars.setSizes([1000, 150])
        self.splitter.addWidget(self.splitter_pars)
        self.splitter.setSizes([1500, 250])

        self.setCentralWidget(self.splitter)

        # >>> create Menu
        #self.initMenu()

        # create toolbar
        # self.toolbar = self.addToolBar('B-spline')
        # self.toolbar.addAction(Bspline)

        #self.draw()
        self.showMaximized()
        self.show()