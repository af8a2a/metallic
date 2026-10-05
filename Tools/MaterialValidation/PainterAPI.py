"""Run inside Painter through its official remote Python endpoint.

Only public substance_painter APIs are used. Bindings are supplied after inspecting
the installed OpenPBR version; ASM/Sheen substitutions are never inferred.
"""
import json
import os
import copy
from pathlib import Path


def dispatch(request):
    import substance_painter.application as app
    import substance_painter.colormanagement as cm
    import substance_painter.export as export
    import substance_painter.js as js
    import substance_painter.layerstack as layers
    import substance_painter.project as project
    import substance_painter.resource as resource
    import substance_painter.textureset as ts

    def shaders():
        return js.evaluate('alg.shaders.shaderInstancesToObject()')

    def inventory():
        result = {'painterVersion': app.version(), 'versionInfo': list(app.version_info()),
                  'projectOpen': project.is_open(), 'channels': list(ts.ChannelType.__members__),
                  'resources': [r.identifier().url() for r in resource.search('OpenPBR')],
                  'exportPresets': [{'name': p.name, 'url': p.url} for p in export.list_predefined_export_presets()],
                  'ocioEnvironment': os.environ.get('OCIO')}
        if project.is_open():
            result.update(projectPath=project.file_path(), needsSaving=project.needs_saving(),
                          ready=project.is_in_edition_state() and not project.is_busy())
            if result['ready']:
                result['shaders'] = shaders()
                result['shaderParameters'] = {str(s['id']): js.evaluate(f'alg.shaders.parameters({s["id"]})')
                                               for s in js.evaluate('alg.shaders.instances()')}
                result['textureSets'] = {t.name(): [c.name for c in t.get_stack().all_channels()]
                                         for t in ts.all_texture_sets()}
        return result

    action = request['action']
    if action == 'inventory':
        return inventory()
    if tuple(app.version_info())[:2] < (12, 1):
        raise RuntimeError('Painter 12.1+ is required; refusing ASM as an OpenPBR reference.')
    if action == 'ready':
        return {'ready': project.is_open() and project.is_in_edition_state() and not project.is_busy()}

    root = Path(request['root']).resolve()
    manifest = json.loads((root/'Manifest.json').read_text(encoding='utf-8'))
    case = next(c for c in manifest['cases'] if c['id'] == request['case'])
    mode = request['mode']
    if mode not in case['modes']:
        raise ValueError('Invalid case mode')
    output = root/case['id']/mode
    spp = output/'source'/f'{case["id"]}_{mode}.spp'
    profile = request.get('profile', {})

    if action == 'open':
        if project.is_open():
            raise RuntimeError('Close the current saved project before opening another')
        if not spp.is_file():
            raise FileNotFoundError(str(spp))
        project.open(str(spp))
        return {'status':'opened_wait_for_idle','path':str(spp)}

    if action == 'verify_reloaded':
        import substance_painter.display as display
        if not project.is_in_edition_state() or project.is_busy() or Path(project.file_path() or '').resolve()!=spp.resolve():
            raise RuntimeError('Expected saved project is not ready')
        receipt=json.loads((output/'AuthoringReceipt.json').read_text(encoding='utf-8'))
        actual_shaders=shaders()
        if actual_shaders!=receipt['shaderReadback']:
            raise ValueError('Reloaded shader definitions differ from authoring readback')
        sets={t.name():t for t in ts.all_texture_sets()}
        if set(sets)!=set(receipt['materials']):
            raise ValueError('Reloaded texture sets differ')
        checks=0
        textures=[]
        for material,parameters in receipt['materials'].items():
            fills=[n for n in layers.get_root_layer_nodes(sets[material].get_stack())
                   if n.get_name()=='Validation constants - ACEScg / raw data']
            if not fills:
                raise ValueError('Missing validation fill layer: '+material)
            fill=fills[0]
            for key,item in parameters.items():
                if 'shaderParameter' in item:
                    continue
                source=fill.get_source(getattr(ts.ChannelType,item['channel']))
                if 'texture' in item:
                    rid=source.resource_id
                    if not resource.Resource.retrieve(rid):
                        raise ValueError('Embedded texture unavailable: '+rid.url())
                    if str(source.get_color_space())!=item['colorSpace']:
                        raise ValueError('Reloaded texture color space differs: '+key)
                    textures.append(rid.url())
                else:
                    vector=isinstance(item['requested'],list)
                    color=source.get_color()
                    values=color.working if vector and not key.startswith('diagnostic_') else color.value
                    if max(abs(a-b) for a,b in zip(values,item['readback']))>1e-5:
                        raise ValueError('Reloaded constant differs: '+material+'/'+key)
                checks+=1
        environment=display.get_environment_resource()
        if not environment or not resource.Resource.retrieve(environment):
            raise ValueError('Embedded environment unavailable')
        return {'case':case['id'],'mode':mode,'passed':True,'sourceChecks':checks,
                'embeddedTextures':textures,'environment':environment.url(),
                'shaderDefinitionsEqual':True,'needsSaving':project.needs_saving()}

    if action == 'probe':
        if project.is_open():
            raise RuntimeError('A project is open. Save and close it before creating a probe.')
        ocio = Path(request['ocio']).resolve()
        if not ocio.is_file():
            raise ValueError('OCIO config is missing')
        old_ocio = os.environ.get('OCIO')
        try:
            os.environ['OCIO'] = str(ocio)
            project.create(str(root/case['mesh']), template_file_path=request['template'],
                           settings=project.Settings(normal_map_format=project.NormalMapFormat.OpenGL,
                           mesh_unit_scale=100.0, default_texture_resolution=512))
        finally:
            if old_ocio is None:
                os.environ.pop('OCIO', None)
            else:
                os.environ['OCIO'] = old_ocio
        return {'status': 'probe_created_no_parameters_authored'}

    if action == 'close':
        if not project.is_open() or Path(project.file_path() or '').resolve() != spp.resolve() or project.needs_saving():
            raise RuntimeError('Refusing to close an unrelated or unsaved project')
        project.close()
        return {'status': 'closed_saved_owned_project'}

    if action == 'remove_duplicate_validation_layers':
        if Path(project.file_path() or '').resolve()!=spp.resolve():
            raise RuntimeError('Unexpected project')
        removed={}
        for texture_set in ts.all_texture_sets():
            nodes=[n for n in layers.get_root_layer_nodes(texture_set.get_stack())
                   if n.get_name()=='Validation constants - ACEScg / raw data']
            for node in nodes[1:]:
                layers.delete_node(node)
            removed[texture_set.name()]=max(0,len(nodes)-1)
        return {'removedOlderDuplicateLayers':removed}

    if action == 'reference':
        import substance_painter.ui as ui
        import substance_painter.display as display
        if Path(project.file_path() or '').resolve()!=spp.resolve():
            raise RuntimeError('Unexpected project for reference')
        ui.show_main_window()
        camera=display.Camera.get_default_camera()
        mode_name=request.get('referenceMode','Edition')
        if mode_name not in ['Edition','Visualisation']:
            raise ValueError(mode_name)
        ui.switch_to_mode(getattr(ui.UIMode,mode_name))
        directory=output/'reference'
        directory.mkdir(parents=True,exist_ok=True)
        camera_file=directory/'Camera.json'
        if mode_name=='Edition' and not camera_file.exists():
            camera.position=[v*1.7 for v in camera.position]
        state={'status':'reference_mode_selected','mode':mode_name,'camera':{
            'position':camera.position,'rotation':camera.rotation,'fieldOfView':camera.field_of_view}}
        if not camera_file.exists():
            camera_file.write_text(json.dumps(state,indent=2),encoding='utf-8')
        return state

    if action == 'capture':
        import substance_painter.ui as ui
        if Path(project.file_path() or '').resolve()!=spp.resolve():
            raise RuntimeError('Unexpected project for capture')
        directory=output/'reference'
        directory.mkdir(parents=True,exist_ok=True)
        mode=ui.get_current_mode()
        name='PainterIrayWindow.png' if mode==ui.UIMode.Visualisation else 'PainterViewportWindow.png'
        destination=directory/name
        if destination.exists():
            raise FileExistsError(str(destination))
        # Public Painter UI entry point and public Qt widget capture. This records
        # the actual app display, not a synthesized material render or linear EXR.
        window=ui.get_main_window()
        if not window.screen().grabWindow(window.winId()).save(str(destination)):
            raise RuntimeError('Qt failed to save the Painter window reference')
        return {'status':'app_window_reference_saved','path':str(destination),'linearRadiance':False}

    if profile.get('verifiedPainterVersion') != app.version():
        raise ValueError('Profile must be verified against this exact Painter version first')
    bindings = profile['bindings']
    required = set().union(*(v['parameters'] for v in case['variants']))
    if mode == 'textured':
        required.update(case['texturedOverrides'])
    missing = sorted(k for k in required if not bindings.get(k))
    if missing:
        raise ValueError('Unverified bindings: ' + ', '.join(missing))
    for key in required:
        binding = bindings[key]
        if (binding.get('channel') not in ts.ChannelType.__members__ and not binding.get('shaderParameter')) or not binding.get('evidence'):
            raise ValueError('Missing channel/semantic evidence: '+key)
    channels = [bindings[k]['channel'] for k in required if 'channel' in bindings[k]]
    if len(channels) != len(set(channels)):
        raise ValueError('Two semantic inputs alias the same Painter channel')

    if action == 'create':
        if project.is_open():
            raise RuntimeError('Save and close the current project before a batch')
        if output.exists():
            raise FileExistsError('Output already exists; choose a fresh prepared package')
        template = profile['templates'].get(case['id']) or profile['templates'].get('default')
        if not template or not Path(template).is_file():
            raise ValueError('Missing verified OpenPBR template')
        if not profile.get('ocioConfig') or not Path(profile['ocioConfig']).is_file():
            raise ValueError('Missing OCIO config')
        if any('renderState' in v for v in case['variants']) and not profile.get('coverageEvidence'):
            raise ValueError('MASK cutoff/double-sided shader state must be verified first')
        return dispatch(dict(request, action='probe', template=template, ocio=profile['ocioConfig']))

    if not project.is_open() or not project.is_in_edition_state() or project.is_busy():
        raise RuntimeError('Painter project is not ready')
    acescg = profile.get('acescgColorSpace', 'ACEScg')
    for color in [(1.,0.,0.), (0.,1.,0.), (0.,0.,1.), (.18,.18,.18)]:
        actual = cm.Color(*color, acescg).working
        if max(abs(a-b) for a,b in zip(actual, color)) > 2e-5:
            raise ValueError('Working RGB is not ACEScg')
    instances = js.evaluate('alg.shaders.instances()')
    if not instances or any('openpbr' not in (s['shader']+' '+s['url']).lower().replace('-','') for s in instances):
        raise ValueError('Active shader is not OpenPBR')
    texture_sets = {t.name(): t for t in ts.all_texture_sets()}
    if set(texture_sets) != {v['id'] for v in case['variants']}:
        raise ValueError('Imported material names differ from the manifest')

    if action == 'author':
        template=profile['templates'].get(case['id']) or profile['templates'].get('default')
        if project.file_path() and Path(project.file_path()).resolve()!=Path(template).resolve():
            raise RuntimeError('Refusing to author an existing saved project')
        if any(n.get_name()=='Validation constants - ACEScg / raw data'
               for t in texture_sets.values() for n in layers.get_root_layer_nodes(t.get_stack())):
            raise RuntimeError('Partial authoring already exists; do not stack a retry over it')
        shader_state=shaders()
        prototype=next(iter(shader_state['shaders'].values()))
        shader_state['shaders']={}
        expected_shaders={}
        for var in case['variants']:
            label='Validation_'+var['id']
            definition=copy.deepcopy(prototype)
            definition['shaderInstance']=label
            values=profile['shaderDefaults'].copy()
            for key,value in var['parameters'].items():
                binding=bindings[key]
                if binding.get('shaderParameter'):
                    values[binding['shaderParameter']]=value if isinstance(value,bool) else value*binding.get('scale',1.)
            params=var['parameters']
            values.update(anisotropyEnabled=params.get('specular_roughness_anisotropy',0)>0,
                          coatEnabled=params.get('coat_weight',0)>0 or (mode=='textured' and 'coat_weight' in case['texturedOverrides']),
                          sheenEnabled=params.get('fuzz_weight',0)>0 or (mode=='textured' and 'fuzz_weight' in case['texturedOverrides']),
                          transmissiveEnabled=params.get('transmission_weight',0)>0,
                          absorptionEnabled='transmission_color' in params)
            if 'renderState' in var:
                values.update(alpha_test_enabled=True,alpha_test_threshold=var['renderState']['alphaCutoff'],doubleSided=True)
            for key,value in values.items():
                groups=[g for g in definition['parameters'].values() if key in g]
                if len(groups)!=1:
                    raise ValueError('Unknown/ambiguous shader parameter: '+key)
                groups[0][key]=value
            shader_state['shaders'][label]=definition
            shader_state['texturesets'][var['id']]['shader']=label
            expected_shaders[label]=values
        js.evaluate('alg.shaders.shaderInstancesFromObject('+json.dumps(shader_state)+')')
        actual_shaders=shaders()
        for label,values in expected_shaders.items():
            flattened={k:v for group in actual_shaders['shaders'][label]['parameters'].values() for k,v in group.items()}
            for key,value in values.items():
                if abs(float(flattened[key])-float(value))>1e-5:
                    raise ValueError('Shader readback mismatch: '+label+'/'+key)
        receipt = {'status':'authored_not_render_validated', 'painterVersion':app.version(),
                   'workingColorSpace':acescg, 'case':case['id'], 'mode':mode, 'materials':{},
                   'profile':profile, 'sourceManifest':manifest['status'],'shaderReadback':actual_shaders}
        for var in case['variants']:
            stack = texture_sets[var['id']].get_stack()
            fill = layers.insert_fill(layers.InsertPosition.from_textureset_stack(stack))
            fill.set_name('Validation constants - ACEScg / raw data')
            seen = {}
            for key,value in var['parameters'].items():
                binding=bindings[key]
                if binding.get('shaderParameter'):
                    seen[key]={'requested':value,'shaderParameter':binding['shaderParameter'],'scale':binding.get('scale',1.)}
                    continue
                channel=getattr(ts.ChannelType,binding['channel'])
                is_vector=isinstance(value,list)
                is_color=is_vector and not key.startswith('diagnostic_')
                if not stack.has_channel(channel):
                    stack.add_channel(channel,ts.ChannelFormat.RGB32F if is_vector else ts.ChannelFormat.L32F)
                stack.edit_channel(channel,ts.ChannelFormat.RGB32F if is_vector else ts.ChannelFormat.L32F)
                scaled=[float(v)*binding.get('scale',1.0) for v in (value if is_vector else [value]*3)]
                source=fill.set_source(channel,cm.Color(*scaled, acescg if is_color else cm.GenericColorSpace.Raw))
                readback=source.get_color().working if is_color else source.get_color().value
                if max(abs(a-b) for a,b in zip(scaled,readback))>1e-5:
                    raise ValueError(f'{key} was clamped or transformed: {readback}, expected {scaled}')
                seen[key]={'requested':value,'readback':list(readback),'channel':channel.name}
            if mode=='textured':
                for key,texture_name in case['texturedOverrides'].items():
                    channel=getattr(ts.ChannelType,bindings[key]['channel'])
                    if not stack.has_channel(channel):
                        stack.add_channel(channel,ts.ChannelFormat.RGB32F)
                    tex=manifest['textures'][texture_name]
                    rid=resource.import_project_resource(str(root/tex['path']),resource.Usage.TEXTURE).identifier()
                    source=fill.set_source(channel,rid)
                    space={'ACEScg':acescg,'data':cm.DataColorSpace.Data,'normal_opengl':cm.NormalColorSpace.NormalXYZRight}[tex['colorSpace']]
                    if channel==ts.ChannelType.Tangent:
                        space=cm.NormalColorSpace.NormalXYZRight
                    source.set_color_space(space)
                    if source.get_color_space()!=space:
                        raise ValueError('Texture color-space override did not stick')
                    seen[key]={'texture':tex,'channel':channel.name,'colorSpace':str(space)}
            receipt['materials'][var['id']]=seen
        if profile.get('environment'):
            import substance_painter.display as display
            environment=resource.import_project_resource(profile['environment'],resource.Usage.ENVIRONMENT).identifier()
            display.set_environment_resource(environment)
            receipt['environment']=environment.url()
        output.mkdir(parents=True,exist_ok=True)
        (output/'AuthoringReceipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        return {'status':'authored_wait_for_idle'}

    if action == 'save':
        if spp.exists() and Path(project.file_path() or '').resolve()!=spp.resolve():
            raise FileExistsError(str(spp))
        if not (output/'AuthoringReceipt.json').is_file():
            raise RuntimeError('No authoring receipt')
        if spp.exists():
            project.save()
        else:
            project.save_as(str(spp))
        return {'status':'saved','path':str(spp)}

    if action == 'export':
        if Path(project.file_path() or '').resolve()!=spp.resolve():
            raise RuntimeError('Unexpected current project')
        preset=profile.get('exportPreset')
        if not preset or not profile.get('exportColorEvidence'):
            raise ValueError('An ACEScg/raw export preset must be numerically verified first')
        destination=output/'textures'
        if destination.exists():
            raise FileExistsError(str(destination))
        config=dict(exportPath=str(destination),exportShaderParams=True,defaultExportPreset=preset,
                    exportList=[{'rootPath':t.name()} for t in texture_sets.values()],
                    exportParameters=[{'parameters':{'fileFormat':'exr','bitDepth':'32f','dithering':False,
                                                     'paddingAlgorithm':'infinite','dilationDistance':16}}])
        planned=export.list_project_textures(config)
        if not planned:
            raise ValueError('Empty export plan')
        result=export.export_project_textures(config)
        if result.status != export.ExportStatus.Success:
            raise RuntimeError(str(result.message))
        evidence={'status':'textures_exported_not_render_validated','files':{str(k):v for k,v in result.textures.items()},
                  'config':config,'shaders':shaders(),'parameterReadback':'AuthoringReceipt.json'}
        (output/'ExportReceipt.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        return evidence
    raise ValueError('Unknown action: '+action)


def safe_dispatch(request):
    import traceback
    try:
        return {'ok':True,'result':dispatch(request)}
    except Exception:
        return {'ok':False,'error':traceback.format_exc()}
