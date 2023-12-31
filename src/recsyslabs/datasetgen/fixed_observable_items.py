
from recsyslabs.datasetgen.tabular_data import TabularData


class TabularDataFixedObservableItems(TabularData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


