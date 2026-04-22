import scvi


def choose_batch_key(adata, candidates: list[str]) -> str:
    for key in candidates:
        if key in adata.obs.columns:
            return key
    raise KeyError(f"None of batch key candidates found in adata.obs: {candidates}")


def train_scvi_and_embed(
    adata,
    batch_key: str,
    n_latent: int = 30,
    max_epochs: int = 100,
):
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
    vae = scvi.model.SCVI(adata, n_latent=n_latent, gene_likelihood="nb")
    vae.train(max_epochs=max_epochs, early_stopping=True)
    z = vae.get_latent_representation()
    return vae, z
