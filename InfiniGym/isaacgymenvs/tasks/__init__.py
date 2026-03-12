
from .fetch.fetch_base import FetchBase
from .fetch.fetch_ptd import FetchPointCloudBase
from .fetch.fetch_naive import FetchNaive

# CuRobo
from .fetch.fetch_mesh_curobo import FetchMeshCurobo
from .fetch.repeat.fetch_mesh_curobo_rep import FetchMeshCuroboRep
from .fetch.fetch_ptd_curobo import FetchPtdCurobo
from .fetch.repeat.fetch_ptd_curobo_rep import FetchPtdCuroboRep

from .fetch.fetch_mesh_curobo_datagen import FetchCuroboDataGen

# Imit Beta
from .fetch.imit.fetch_ptd_imit_e2e import FetchPtdImitE2E
from .fetch.imit.fetch_ptd_imit_two_stage import FetchPtdImitTwoStage
from .fetch.imit.fetch_ptd_imit_curobo_cgn import FetchPtdImitCuroboCGN

# GO
from .fetch.fetch_mesh_curobo_go import FetchMeshCuroboGO

isaacgym_task_map = {
    "FetchBase": FetchBase,
    "FetchPointCloudBase": FetchPointCloudBase,

    "FetchNaive": FetchNaive,
    "FetchMeshCurobo": FetchMeshCurobo,
    "FetchMeshCuroboRep": FetchMeshCuroboRep,

    "FetchPtdCurobo": FetchPtdCurobo,
    "FetchPtdCuroboRep": FetchPtdCuroboRep,

    "FetchCuroboDataGen": FetchCuroboDataGen,

    "FetchPtdImitE2E": FetchPtdImitE2E,
    "FetchPtdImitTwoStage": FetchPtdImitTwoStage,
    "FetchPtdImitCuroboCGN": FetchPtdImitCuroboCGN,
    
    "FetchMeshCuroboGO": FetchMeshCuroboGO,
}


try:
    # Contact_Graspnet_Pytorch
    from .fetch.fetch_mesh_curobo_cgn_beta import FetchMeshCuroboPtdCGNBeta

    isaacgym_task_map = {
        "FetchBase": FetchBase,
        "FetchPointCloudBase": FetchPointCloudBase,

        "FetchNaive": FetchNaive,
        "FetchMeshCurobo": FetchMeshCurobo,
        "FetchMeshCuroboRep": FetchMeshCuroboRep,

        "FetchPtdCurobo": FetchPtdCurobo,
        "FetchPtdCuroboRep": FetchPtdCuroboRep,

        "FetchCuroboDataGen": FetchCuroboDataGen,

        "FetchPtdImitE2E": FetchPtdImitE2E,
        "FetchPtdImitTwoStage": FetchPtdImitTwoStage,
        "FetchPtdImitCuroboCGN": FetchPtdImitCuroboCGN,
        
        "FetchMeshCuroboPtdCGNBeta": FetchMeshCuroboPtdCGNBeta,
        
        "FetchMeshCuroboGO": FetchMeshCuroboGO,
    }

except:
    print("============================================================")
    print("Import Error: Additional Method Excluded.")


try:
    # OMPL
    from .fetch.fetch_mesh_pyompl import FetchMeshPyompl
    from .fetch.repeat.fetch_mesh_pyompl_rep import FetchMeshPyomplRep
    from .fetch.fetch_ptd_pyompl import FetchPtdPyompl
    from .fetch.repeat.fetch_ptd_pyompl_rep import FetchPtdPyomplRep

    # Cabinet
    from .fetch.fetch_ptd_cabinet import FetchPtdCabinet
    from .fetch.fetch_ptd_cabinet_cgn_beta import FetchPtdCabinetCGNBeta

    # Contact_Graspnet_Pytorch
    from .fetch.fetch_ptd_curobo_cgn_beta import FetchPtdCuroboCGNBeta
    from .fetch.fetch_ptd_pyompl_cgn_beta import FetchPtdPyomplCGNBeta
    from .fetch.fetch_mesh_curobo_cgn_beta import FetchMeshCuroboPtdCGNBeta
    from .fetch.fetch_mesh_pyompl_cgn_beta import FetchMeshPyomplPtdCGNBeta

    from .fetch.repeat.fetch_mesh_curobo_cgn_beta_rep import FetchMeshCuroboPtdCGNBetaRep
    from .fetch.repeat.fetch_ptd_curobo_cgn_beta_rep import FetchPtdCuroboCGNBetaRep
    from .fetch.repeat.fetch_ptd_pyompl_cgn_beta_rep import FetchPtdPyomplCGNBetaRep

    isaacgym_task_map = {
        "FetchBase": FetchBase,
        "FetchPointCloudBase": FetchPointCloudBase,

        "FetchNaive": FetchNaive,
        "FetchMeshCurobo": FetchMeshCurobo,
        "FetchMeshPyompl": FetchMeshPyompl,
        "FetchMeshCuroboRep": FetchMeshCuroboRep,
        "FetchMeshPyomplRep": FetchMeshPyomplRep,

        "FetchPtdCurobo": FetchPtdCurobo,
        "FetchPtdPyompl": FetchPtdPyompl,

        "FetchCuroboDataGen": FetchCuroboDataGen,

        "FetchPtdCuroboRep": FetchPtdCuroboRep,
        "FetchPtdPyomplRep": FetchPtdPyomplRep,

        "FetchPtdCuroboCGNBeta": FetchPtdCuroboCGNBeta,
        "FetchPtdPyomplCGNBeta": FetchPtdPyomplCGNBeta,
        "FetchMeshCuroboPtdCGNBeta": FetchMeshCuroboPtdCGNBeta,
        "FetchMeshPyomplPtdCGNBeta": FetchMeshPyomplPtdCGNBeta,

        "FetchPtdCabinet": FetchPtdCabinet,
        "FetchPtdCabinetCGNBeta": FetchPtdCabinetCGNBeta,

        "FetchMeshCuroboPtdCGNBetaRep": FetchMeshCuroboPtdCGNBetaRep,
        "FetchPtdCuroboCGNBetaRep": FetchPtdCuroboCGNBetaRep,
        "FetchPtdPyomplCGNBetaRep": FetchPtdPyomplCGNBetaRep,

        "FetchPtdImitE2E": FetchPtdImitE2E,
        "FetchPtdImitTwoStage": FetchPtdImitTwoStage,
        "FetchPtdImitCuroboCGN": FetchPtdImitCuroboCGN,
        
        "FetchMeshCuroboGO": FetchMeshCuroboGO,
    }

except Exception as e:    # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
    print("============================================================")
    print("Import Error: Additional Method Excluded.", e)

