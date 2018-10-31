from functools import partial
from matplotlib import cm
from PyQt5.QtCore import (Qt, )
from PyQt5.QtGui import (QFont)
from PyQt5.QtWidgets import (QApplication, QMessageBox, QMainWindow, QSplitter, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QPushButton, QHeaderView, QCheckBox,
                             QRadioButton, QButtonGroup, QComboBox)
import pyqtgraph as pg

from H2_exc import *

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
        pg.TextItem.__init__(self, text=text, fill=pg.mkBrush(0, 0, 0, 0))
        self.setFont(QFont("SansSerif", 16))
        self.setPos(x, y)
        self.redraw()

    def redraw(self):
        if self.active:
            self.setColor((255, 225, 53))
        else:
            self.setColor((255, 255, 255))
        self.parent.parent.plot_exc.add_model(self.name, add=self.active)

    def mouseClickEvent(self, ev):

        if ev.double():
            self.active = not self.active
            self.redraw()

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
        self.s_status = False
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
                self.mousePoint = self.vb.mapSceneToView(event.pos())
                r = self.vb.viewRange()
                self.ind = np.argmin(((self.mousePoint.x() - self.data['N']) / (r[0][1] - r[0][0]))**2   + ((self.mousePoint.y() - self.data['b']) / (r[1][1] - r[1][0]))**2)
                if self.selected_point is not None:
                    self.vb.removeItem(self.selected_point)
                self.selected_point = pg.ScatterPlotItem(x=[self.data['N'][self.ind]], y=[self.data['b'][self.ind]], symbol='o', size=15,
                                                        pen={'color': 0.8, 'width': 1}, brush=pg.mkBrush(230, 100, 10))
                self.vb.addItem(self.selected_point)
                N = [float(self.parent.item(i,1).text()) for i in range(self.parent.rowCount())]
                b = [float(self.parent.item(i,3).text()) for i in range(self.parent.rowCount())]
                ind = np.argmin((b - self.data['b'][self.ind])**2 + (N - self.data['N'][self.ind])**2)
                #ind = np.where(np.logical_and(b == self.data['b'][self.ind], N == self.data['N'][self.ind]))[0][0]
                #ind = np.where(np.logical_and(self.parent.data['b'] == self.data['b'][self.ind], self.parent.data['N'] == self.data['N'][self.ind]))[0][0]
                self.parent.setCurrentCell(0, 0)
                self.parent.row_clicked(ind)

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
                self.s_status = False

class QSOlistTable(pg.TableWidget):
    def __init__(self, parent):
        super().__init__(editable=False, sortable=False)
        self.setStyleSheet(open('styles.ini').read())
        self.parent = parent
        self.format = None

        self.contextMenu.addSeparator()
        self.contextMenu.addAction('H2 compare').triggered.connect(self.compare)

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
        pars = [list(grid.keys())[list(grid.values()).index('x')],
                list(grid.keys())[list(grid.values()).index('y')]]
        fixed = list(grid.keys())[list(grid.values()).index('fixed')]
        fixed = {fixed: float(getattr(self.parent.parent.grid_pars, fixed + '_val').currentText())}
        for idx in self.selectedIndexes():
            # self.parent.normview = False
            name = self.cell_value('name')
            self.parent.parent.H2.comparegrid(name, pars=pars, fixed=fixed, syst=0.2, plot=False)
            grid = self.parent.parent.H2.grid
            #print('grid', grid['uv'], grid['n0'], grid['lnL'])
            self.parent.parent.plot_reg.set_data(x=grid[pars[0]], y=grid[pars[1]], z=grid['lnL'])
            self.parent.parent.plot_reg.setLabels(bottom='log('+pars[0]+')', left='log('+pars[1]+')')
            #self.pos = [self.x[0] - (self.x[1] - self.x[0]) / 2, self.y[0] - (self.y[1] - self.y[0]) / 2]
            #self.scale = [(self.x[-1] - self.x[0]) / (self.x.shape[0] - 1), (self.y[-1] - self.y[0]) / (self.y.shape[0] - 1)]

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
        print(data)
        self.table.setdata(data)
        self.buttons = {}
        for i, d in enumerate(data):
            wdg = QWidget()
            l = QVBoxLayout()
            l.addSpacing(3)
            button = QPushButton(d[0], self, checkable=True)
            button.setFixedSize(100, 20)
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
        self.pars = {'uv': 'x', 'n0': 'y', 'me': 'fixed'}
        self.parent.H2.setgrid(pars=list(self.pars.keys()))

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
            getattr(self, n + '_val').setFixedSize(80,30)
            getattr(self, n + '_val').addItems(np.append([''], np.asarray(np.sort(np.unique(self.parent.H2.grid[n])), dtype=str)))
            if self.pars[n] is 'fixed':
                getattr(self, n + '_val').setCurrentIndex(1)
            l.addWidget(getattr(self, n + '_val'))
            l.addStretch(1)
            layout.addLayout(l)

        layout.addStretch(1)

        self.setLayout(layout)

        self.setStyleSheet(open('styles.ini').read())

    def setGridView(self, par, b):
        self.pars[par] = b
        if b is 'fixed':
            print(getattr(self, par + '_val').text())
            self.pars[par] = getattr(self, par + '_val').currentText()
        print(self.pars)

class H2viewer(QMainWindow):

    def __init__(self):
        super().__init__()

        self.H2 = H2_exc(folder='data/')
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