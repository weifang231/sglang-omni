# SPDX-License-Identifier: Apache-2.0
class AlwaysAdmit:
    def admit(self, request_id, request):
        return True

    def completed(self, request_id):
        pass

    aborted = completed


class RejectAll(AlwaysAdmit):
    def admit(self, request_id, request):
        return False


def always_admit(*, config):
    return AlwaysAdmit()


def reject_all(*, config):
    return RejectAll()
