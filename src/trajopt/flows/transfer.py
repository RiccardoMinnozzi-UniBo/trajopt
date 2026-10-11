"""Check physical channel compatibility before importing a stage's guesses."""


def phase_layouts(trajectory, config):
    phases = trajectory.phases if hasattr(trajectory, 'phases') else {'main': trajectory}
    configs = config.phases if hasattr(trajectory, 'phases') else {'main': config}
    layouts = {}
    for name, phase in phases.items():
        layout = {}
        for kind in ('state', 'control'):
            groups = phase.index_map.components[kind]
            layout[kind] = (
                getattr(phase.index_map.n, kind),
                tuple(sorted((key, tuple(int(i) for i in indices),
                              configs[name][kind][key].get('units', configs[name][kind][key].get('unit')))
                             for key, indices in groups.items())),
            )
        time = configs[name].get('time', {})
        layout['time_unit'] = time.get('units', time.get('unit'))
        layouts[name] = layout
    return layouts


def transfer_guesses(source, target_layouts, phases=None):
    selected = tuple(source.trajectory) if phases is None else tuple(phases)
    if not selected:
        raise ValueError(f"Stage '{source.name}' has no trajectory to transfer")
    guesses = {}
    for name in selected:
        if name not in source.trajectory:
            raise ValueError(f"Source stage '{source.name}' has no phase '{name}'")
        if name not in target_layouts:
            raise ValueError(f"Destination has no phase '{name}'")
        if source.layouts[name] != target_layouts[name]:
            raise ValueError(f"Phase '{name}' has incompatible channel identities, ordering, or units")
        guesses[name] = source.trajectory[name]
    return guesses
