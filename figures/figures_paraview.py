import paraview.simple as pvs

from postprocess import pvlib


def rae_mesh():
    filename = "/scratchm/pseize/RAE_2822/MF/RUN_1/ENSIGHT/archive_CHARME.volu.ins.case"
    plotter = pvlib.Plotter()
    reader, _, _ = plotter.load_data(filename, [])

    view1 = pvs.CreateRenderView(InteractionMode='2D')
    view2 = pvs.CreateRenderView(InteractionMode='2D')
    view3 = pvs.CreateRenderView(InteractionMode='2D')

    pvs.Show(reader, view1, Representation='Wireframe')
    view1.CameraPosition = [1, 0.25, 1]
    view1.CameraFocalPoint = [1, 0.25, 0]
    view1.CameraParallelScale = 0.5

    pvs.Show(reader, view2, Representation='Wireframe')
    view2.CameraPosition = [0.02, 0.01, 1]
    view2.CameraFocalPoint = [0.02, 0.01, 0]
    view2.CameraParallelScale = 0.05
    view2.OrientationAxesVisibility = 0

    pvs.Show(reader, view3, Representation='Wireframe')
    view3.CameraPosition = [0.6, 0.25, 1]
    view3.CameraFocalPoint = [0.6, 0.25, 0.0]
    view3.CameraParallelScale = 0.25
    view3.OrientationAxesVisibility = 0

    layout = pvs.CreateLayout()
    id_1 = layout.SplitVertical(0, 0.4)
    id_2 = layout.SplitHorizontal(id_1 + 1, 0.5)
    layout.AssignView(id_1, view1)
    layout.AssignView(id_2, view2)
    layout.AssignView(id_2 + 1, view3)
    layout.SetSize(1920, 850)

    print("Please set the OrientationAxis manually in view1")
    view1.OrientationAxesInteractivity = 1
    plotter.draw(0, True)
    # pvs.SaveScreenshot("rae_mesh.png", layout)


def rae_field():
    filename = "/scratchm/pseize/RAE_2822/MF/RUN_1/ENSIGHT/archive_CHARME.volu.ins.case"
    plotter = pvlib.Plotter()
    reader, view, _ = plotter.load_data(filename, ['P', 'Mach'], render_view=True, rvs=(1200, 600))
    display = pvs.Show(reader, view, ColorArrayName=['CELLS', 'P'])
    color_bar = pvs.GetScalarBar(pvs.GetColorTransferFunction('P'), view)
    color_bar.WindowLocation = 'Lower Center'
    color_bar.Title = r'$P \quad \left( \operatorname{Pa} \right)$'
    color_bar.ComponentTitle = ''
    color_bar.Orientation = 'Horizontal'
    color_bar.TitleFontSize = 20
    color_bar.LabelFontSize = 15
    color_bar.ScalarBarLength = 0.6
    color_bar.RangeLabelFormat = '%.0f'
    view.CameraPosition = [0.5, 0, 1]
    view.CameraFocalPoint = [0.5, 0, 0]
    view.CameraParallelScale = 1
    plotter.scale_display(display)

    c2p = pvs.CellDatatoPointData(reader, ProcessAllArrays=0, CellDataArraytoprocess=['Mach'])
    ctr = pvs.Contour(c2p, ContourBy=['POINTS', 'Mach'], Isosurfaces=[1])
    pvs.Show(ctr, view, Representation='Wireframe', LineWidth=3, ColorArrayName=['CELLS', None])

    plotter.draw(0, block=True)
    # pvs.SaveScreenshot("rae_field.png", view)


def rae_cp():
    plotter = pvlib.CpPlotter(view_size=(800, 600), xlabel='$x/c$', ylabel='',
                              title=r'$C_p = \frac{{ P - P_\infty }}'
                                    r'{{ \frac{{1}}{{2}} P_\infty \gamma \operatorname{{Ma}}^2 }} $')
    plotter.view.ChartTitleFontSize = 24
    plotter.view.LegendFontSize = 24
    plotter.register_plot("/scratchm/pseize/RAE_2822/BASE/RUN_1/ENSIGHT/archive_CHARME.surf.ins.case",
                          p_inf=26500, gamma=1.4, mach=0.75, block_name=['Intrados', 'Extrados'], label='Base')
    plotter.register_plot("/scratchm/pseize/RAE_2822/MF/RUN_1/ENSIGHT/archive_CHARME.surf.ins.case", marker=1,
                          p_inf=26500, gamma=1.4, mach=0.75, block_name=['Intrados', 'Extrados'], label='MF')
    plotter.draw(duration=0, block=True)
    # pvs.SaveScreenshot("rae_cp.png", plotter.view)


if __name__ == '__main__':
    # rae_mesh()
    # rae_field()
    # rae_cp()
    pass
