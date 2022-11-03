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

    pvs.SaveScreenshot("rae_mesh.png", layout)


if __name__ == '__main__':
    rae_mesh()
