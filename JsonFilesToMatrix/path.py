class Path:
    def __init__(self, app_api, invoke, package, method) -> None:
        self.app_api = app_api
        self.invoke = invoke
        self.package = package
        self.method = method

    def meta_path_1(self):
        return self.app_api @ self.app_api.T

    def meta_path_2(self):
        return self.app_api @ self.method @ self.app_api.T

    def meta_path_3(self):
        return self.app_api @ self.package @ self.app_api.T

    def meta_path_4(self):
        return self.app_api @ self.invoke @ self.app_api.T
