import os.path

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
    view3.CameraFocalPoint = [0.6, 0.25, 0]
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
    color_bar.Title = '$P$ (Pa)'
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


def rae_mesh_fine():
    filename = "/tmp_user/sator/pseize/RAE_2822/MF/RUN_8/ENSIGHT/archive_CHARME.volu.ins.case"
    plotter = pvlib.Plotter()
    reader, _, _ = plotter.load_data(filename, [])

    view1 = pvs.CreateRenderView(InteractionMode='2D')
    view2 = pvs.CreateRenderView(InteractionMode='2D')
    view3 = pvs.CreateRenderView(InteractionMode='2D')

    pvs.Show(reader, view1, Representation='Wireframe')
    view1.CameraPosition = [0.5, -1, 0]
    view1.CameraFocalPoint = [0.5, 0, 0]
    view1.CameraViewUp = [0, 0, 1]
    view1.CameraParallelScale = 0.5

    pvs.Show(reader, view2, Representation='Wireframe')
    view2.CameraPosition = [0, -1, 0]
    view2.CameraFocalPoint = [0, 0, 0]
    view2.CameraViewUp = [0, 0, 1]
    view2.CameraParallelScale = 0.075
    view2.OrientationAxesVisibility = 0

    pvs.Show(reader, view3, Representation='Wireframe')
    view3.CameraPosition = [1, -1, 0]
    view3.CameraFocalPoint = [1, 0, 0]
    view3.CameraViewUp = [0, 0, 1]
    view3.CameraParallelScale = 0.05
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
    # pvs.SaveScreenshot("rae_mesh_fine.png", layout)


def rae_field_fine():
    plotter = pvlib.Plotter()

    views = [pvs.CreateRenderView() for _ in range(4)]

    for file, i in zip(('BASE/RUN_3_1e-4', 'MF/RUN_8'), (0, 2)):
        reader, _, _ = plotter.load_data(
            os.path.join("/tmp_user/sator/pseize/RAE_2822", file, "ENSIGHT/archive_CHARME.volu.ins.case"),
            ['NuTilde', 'Residu~RhoNuTilde~'])
        pvs.Show(reader, views[i]).SetScalarBarVisibility(views[i], True)
        pvs.Show(reader, views[i + 1]).SetScalarBarVisibility(views[i + 1], True)
        threshold = pvs.Threshold(reader, Scalars=['CELLS', 'Residu~RhoNuTilde~'],
                                  ThresholdMethod='Above Upper Threshold', UpperThreshold=1.9414055995803974 - 0.1)
        pvs.Show(threshold, views[i], Representation='Feature Edges', ColorArrayName=None, AmbientColor=[0, 1, 0],
                 LineWidth=3)
        pvs.Show(threshold, views[i + 1], Representation='Feature Edges', ColorArrayName=None, AmbientColor=[0, 1, 0],
                 LineWidth=3)

    for view in views[::2]:
        view.CameraPosition = [-1, -1.5, 1]
        view.CameraFocalPoint = [1, -0.1, -0.5]
        view.CameraViewUp = [0.5, 0.16, 0.85]
        view.CameraParallelScale = 21
        pvs.GetScalarBar(pvs.GetColorTransferFunction('NuTilde'), view).Visibility = 0
        view.OrientationAxesInteractivity = 1
    for view in views[1::2]:
        view.CameraPosition = [-1, 0.3, -1]
        view.CameraFocalPoint = [-0.02, -0.25, -0.25]
        view.CameraViewUp = [0.5, -0.2, -0.85]
        view.CameraParallelScale = 21
        pvs.GetScalarBar(pvs.GetColorTransferFunction('NuTilde'), view).Visibility = 0
        view.OrientationAxesInteractivity = 1

    space_view = pvs.CreateRenderView()
    space_view.UseColorPaletteForBackground = 0
    space_view.Background = [1, 1, 1]
    space = 0.499

    view_bar = pvs.CreateRenderView()
    view_bar.OrientationAxesVisibility = 0
    color_bar = pvs.GetScalarBar(pvs.GetColorTransferFunction('NuTilde'), view_bar)
    color_bar.WindowLocation = 'Lower Center'
    color_bar.Title = r'$\tilde{\nu} \quad \left( \operatorname{m}^2 / \operatorname{s} \right)$'
    color_bar.ComponentTitle = ''
    color_bar.Orientation = 'Horizontal'
    color_bar.TitleFontSize = 24
    color_bar.LabelFontSize = 18
    color_bar.ScalarBarLength = 0.5

    layout = pvs.CreateLayout()
    id_1 = layout.SplitVertical(0, 0.8)
    layout.AssignView(id_1 + 1, view_bar)

    id_1 = layout.SplitHorizontal(id_1, space)
    id_bar = layout.SplitHorizontal(id_1 + 1, (1 - 2 * space) / (1 - space))
    id_1 = layout.SplitVertical(id_1, 0.5)
    id_3 = layout.SplitVertical(id_bar + 1, 0.5)
    layout.AssignView(id_bar, space_view)
    layout.AssignView(id_1, views[0])
    layout.AssignView(id_1 + 1, views[1])
    layout.AssignView(id_3, views[2])
    layout.AssignView(id_3 + 1, views[3])
    layout.SetSize(1157, 512)

    transfert_function = pvs.GetColorTransferFunction('NuTilde')
    transfert_function.RescaleTransferFunction(0, 3e-5)
    transfert_function.EnableOpacityMapping = 1
    pvs.GetOpacityTransferFunction('NuTilde').Points = [3e-5, 1, 0.5, 0, 3e-5, 0, 0.5, 0]

    print("Please set the OrientationAxis manually in all views")
    plotter.draw(0, True)
    # pvs.SaveScreenshot("rae_field_fine.png", layout)


def sphere_fields():
    plotter = pvlib.SpherePlotter()
    reader, _, _ = plotter.load_data(
        "/tmp_user/sator/pseize/SPHERE_LOBB/MF/RUN_2/_ENSIGHT_/archive_CHARME.volu.ins.case", ['P', 'T', 'Y_NO', 'Y_e'])

    view = pvs.CreateRenderView(InteractionMode='2D')
    pvs.Show(reader, view, Representation='Wireframe', ColorArrayName=[])

    views = [view]
    for var, label, fmt in zip(
            ('P', 'T', 'Y_NO', 'Y_e'),
            ('$P$ (Pa)', '$T$ (K)', 'NO mass fraction', 'e mass fraction'),
            ('%.0f', '%.0f', '%.2g', '%.2g')):
        view = pvs.CreateRenderView(InteractionMode='2D')
        view.OrientationAxesVisibility = 0
        display = pvs.Show(reader, view, Representation='Surface', ColorArrayName=['CELLS', var])
        pvs.ColorBy(display, var, separate=True)
        display.RescaleTransferFunctionToDataRange(True, False)
        display.SetScalarBarVisibility(view, True)
        color_bar = pvs.GetScalarBar(pvs.GetColorTransferFunction(var, display, separate=True), view)
        color_bar.WindowLocation = 'Lower Center'
        color_bar.Title = label
        color_bar.Orientation = 'Horizontal'
        color_bar.TitleFontSize = 24
        color_bar.LabelFontSize = 20
        color_bar.ScalarBarLength = 0.5
        color_bar.RangeLabelFormat = fmt
        views.append(view)

    layout = pvs.CreateLayout()
    id_1 = layout.SplitHorizontal(0, 0.33)
    id_2 = layout.SplitHorizontal(id_1 + 1, 0.5)
    id_3 = layout.SplitVertical(id_2 + 1, 0.5)
    id_2 = layout.SplitVertical(id_2, 0.5)
    layout.AssignView(id_1, views[0])
    layout.AssignView(id_2, views[1])
    layout.AssignView(id_2 + 1, views[3])
    layout.AssignView(id_3, views[2])
    layout.AssignView(id_3 + 1, views[4])
    layout.SetSize(1157, 1000)

    views[0].CameraPosition = [-0.0054, 0.004, 1]
    views[0].CameraFocalPoint = [-0.0054, 0.004, 0]
    views[0].CameraParallelScale = 0.0055
    for view in views[1:]:
        view.CameraPosition = [-0.0037, 0.00425, 1]
        view.CameraFocalPoint = [-0.0037, 0.00425, 0]
        view.CameraParallelScale = 0.007

    print("Please set the OrientationAxis manually in the mesh view")
    views[0].OrientationAxesInteractivity = 1
    plotter.draw(0, True)

    # pvs.SaveScreenshot("sphere_fields.png", layout)


def sphere_carbuncle():
    path = "/tmp_user/sator/pseize/SPHERE_LOBB/CARBUNCLE/RUN_{0}/_ENSIGHT_/archive_CHARME.volu.ins.case"
    plotter = pvlib.SpherePlotter()
    plotter.register_plot(path.format('KO'), 'P', stream=10)
    plotter.register_plot(path.format('OK'), 'P', stream=10)
    view1 = plotter.get_views(path.format('KO'))[0]
    view2 = plotter.get_views(path.format('OK'))[0]
    view_bar = pvs.CreateRenderView()

    for view in (view1, view2):
        color_bar = pvs.GetScalarBar(pvs.GetColorTransferFunction('P'), view)
        color_bar.Visibility = 0
        view.CameraPosition = [-0.007, 0.0019, 1]
        view.CameraFocalPoint = [-0.007, 0.0019, 0]
        view.CameraParallelScale = 0.002

    view_bar.OrientationAxesVisibility = 0
    color_bar = pvs.GetScalarBar(pvs.GetColorTransferFunction('P'), view_bar)
    color_bar.WindowLocation = 'Lower Center'
    color_bar.Title = '$P$ (Pa)'
    color_bar.ComponentTitle = ''
    color_bar.Orientation = 'Horizontal'
    color_bar.TitleFontSize = 24
    color_bar.LabelFontSize = 20
    color_bar.ScalarBarLength = 0.7
    color_bar.RangeLabelFormat = '%.0f'
    color_bar.AutomaticLabelFormat = 0
    color_bar.LabelFormat = '%.1E'

    layout = pvs.CreateLayout()
    id_1 = layout.SplitVertical(0, 0.8)
    layout.AssignView(id_1 + 1, view_bar)
    id_1 = layout.SplitHorizontal(id_1, 0.5)
    layout.AssignView(id_1 + 0, view1)
    layout.AssignView(id_1 + 1, view2)
    layout.SetSize(1157, 500)

    print("Please set the OrientationAxis manually in both top views")
    view1.OrientationAxesInteractivity = 1
    view2.OrientationAxesInteractivity = 1
    plotter.draw(0, True)

    pvs.GetAnimationScene().GoToLast()
    # pvs.SaveScreenshot("sphere_carbuncle.png", layout)


def covo_cedre_fields():
    plotter = pvlib.COVOPlotter()
    plotter.register_plot("/visu/pseize/COVO/BASE/RUN_1/_ENSIGHT_/archive_CHARME.volu.ins.case", 'P', contour=50,
                          r_gas=1, label='Base')

    view1 = pvs.CreateRenderView(InteractionMode='2D')
    for k, s in pvs.GetSources().items():
        if 'Contour' in k[0]:
            force_time = pvs.ForceTime(s, ForcedTime=0)
            display = pvs.Show(force_time, view1)
            display.SetScalarBarVisibility(view1, True)
            text = pvs.Text(Text='Initialisation')
            pvs.Show(text, view1, 'TextSourceRepresentation', WindowLocation='Upper Center', Interactivity=0)
        elif 'EnSightReader' in k[0]:
            pvs.Show(s, view1, ColorArrayName=[])
        elif 'EnSightReader' in k[0]:
            pvs.Show(s, view1)

    plotter.register_plot("/visu/pseize/COVO/BASE/RUN_2/_ENSIGHT_/archive_CHARME.volu.ins.case", 'P', contour=50,
                          r_gas=1, label='Base')
    plotter.register_plot("/visu/pseize/COVO/EXP/RUN_1/_ENSIGHT_/archive_CHARME.volu.ins.case", 'P', contour=50,
                          r_gas=1, label='Exponential Rosenbrock-Euler')
    plotter.register_plot("/visu/pseize/COVO/BASE/RUN_RK4/_ENSIGHT_/archive_CHARME.volu.ins.case", 'P', contour=50,
                          r_gas=1, label='RK4')

    view2 = plotter.get_views("/visu/pseize/COVO/EXP/RUN_1/_ENSIGHT_/archive_CHARME.volu.ins.case")[0]
    view3 = plotter.get_views("/visu/pseize/COVO/BASE/RUN_1/_ENSIGHT_/archive_CHARME.volu.ins.case")[0]
    view4 = plotter.get_views("/visu/pseize/COVO/BASE/RUN_2/_ENSIGHT_/archive_CHARME.volu.ins.case")[0]
    view5 = plotter.get_views("/visu/pseize/COVO/BASE/RUN_RK4/_ENSIGHT_/archive_CHARME.volu.ins.case")[0]
    view_bar = pvs.CreateRenderView()

    layout = pvs.CreateLayout()
    id_1 = layout.SplitVertical(0, 0.9)
    layout.AssignView(id_1 + 1, view_bar)
    id_1 = layout.SplitVertical(id_1, 0.5)
    id_3 = layout.SplitHorizontal(id_1 + 1, 0.33)
    id_4 = layout.SplitHorizontal(id_3 + 1, 0.5)
    id_1 = layout.SplitHorizontal(id_1, 0.5)
    layout.AssignView(id_1, view1)
    layout.AssignView(id_1 + 1, view2)
    layout.AssignView(id_3, view3)
    layout.AssignView(id_4, view4)
    layout.AssignView(id_4 + 1, view5)
    layout.SetSize(1900, 1300)

    for view in (view1, view2, view3, view4, view5):
        color_bar = pvs.GetScalarBar(pvs.GetColorTransferFunction('P'), view)
        color_bar.Visibility = 0
        view.CameraPosition = [0.5, 0, 1]
        view.CameraFocalPoint = [0.5, 0, 0]
        view.CameraParallelScale = 12

    view_bar.OrientationAxesVisibility = 0
    color_bar = pvs.GetScalarBar(pvs.GetColorTransferFunction('P'), view_bar)
    color_bar.WindowLocation = 'Lower Center'
    color_bar.Title = r'$P$'
    color_bar.ComponentTitle = ''
    color_bar.Orientation = 'Horizontal'
    color_bar.TitleFontSize = 24
    color_bar.LabelFontSize = 20
    color_bar.ScalarBarLength = 0.5
    color_bar.RangeLabelFormat = '%.4g'

    for k, s in pvs.GetRepresentations().items():
        if 'TextSourceRepresentation' in k[0]:
            s.FontSize = 24

    # plotter.draw(0, block=False)
    pvs.GetAnimationScene().GoToLast()
    pvs.SaveScreenshot("covo_cedre_fields.png", layout)


def tgv_fields():
    pvs.Connect('localhost')

    views = {}
    for n, label in zip((0, 3000, 6000, 9000), ('$t$ = 0', r'$t$ = 6', r'$t$ = 12', '$t$ = 18')):
        views[n] = pvs.CreateRenderView()
        reader = pvs.OpenDataFile('/visu/pseize/TGV/VISU/sol_{0:08d}/sol_{0:08d}.pvtu'.format(n))
        pvs.Show(reader, views[n], Representation='Outline')
        pvs.Show(pvs.Text(Text=label), views[n], 'TextSourceRepresentation', WindowLocation='Upper Left Corner',
                 FontSize=24, FontFamily='File', FontFile='/usr/share/fonts/lyx/cmr10.ttf')
        calc = pvs.Calculator(reader, ResultArrayName='Norm_U', Function='sqrt(u*u + v*v + w*w)')
        ctr = pvs.Contour(calc, ContourBy=['POINTS', 'qcrit'], Isosurfaces=[0.1, ])
        display = pvs.Show(ctr, views[n], ColorArrayName=['POINTS', 'Norm_U'])
        display.SetScalarBarVisibility(views[n], True)

    view_bar = pvs.CreateRenderView()

    layout = pvs.CreateLayout()
    id_1 = layout.SplitVertical(0, 0.9)
    layout.AssignView(id_1 + 1, view_bar)
    id_1 = layout.SplitVertical(id_1, 0.5)
    id_3 = layout.SplitHorizontal(id_1 + 1, 0.5)
    id_1 = layout.SplitHorizontal(id_1, 0.5)
    layout.AssignView(id_1, views[0])
    layout.AssignView(id_1 + 1, views[3000])
    layout.AssignView(id_3, views[6000])
    layout.AssignView(id_3 + 1, views[9000])
    layout.SetSize((1157, 1157))

    transfert_function = pvs.GetColorTransferFunction('qcrit')
    transfert_function.RescaleTransferFunction(0.0, 1.2)
    for n in views:
        view = views[n]
        pvs.GetScalarBar(transfert_function, views[n]).Visibility = 0
        view.CameraPosition = [17.7, 6, 6.2]
        view.CameraFocalPoint = [0.7, -1, -0.8]
        view.CameraViewUp = [0, 0, 1]
        view.CameraParallelScale = 5

    view_bar.OrientationAxesVisibility = 0
    color_bar = pvs.GetScalarBar(transfert_function, view_bar)
    color_bar.WindowLocation = 'Lower Center'
    color_bar.Title = r'Velocity norm'
    color_bar.ComponentTitle = ''
    color_bar.Orientation = 'Horizontal'
    color_bar.TitleFontSize = 24
    color_bar.TitleFontFamily = 'File'
    color_bar.TitleFontFile = '/usr/share/fonts/lyx/cmr10.ttf'
    color_bar.LabelFontSize = 20
    color_bar.LabelFontFamily = 'File'
    color_bar.LabelFontFile = '/usr/share/fonts/lyx/cmr10.ttf'
    color_bar.ScalarBarLength = 0.5
    color_bar.RangeLabelFormat = '%.4g'

    pvs.GetAnimationScene().GoToLast()
    pvs.SaveScreenshot("tgv_fields.png", layout)


if __name__ == '__main__':
    # rae_mesh()
    # rae_field()
    # rae_mesh_fine()
    # rae_field_fine()
    # sphere_fields()
    # sphere_carbuncle()
    # covo_cedre_mesh()
    # covo_cedre_fields()
    # tgv_fields()
    pass
