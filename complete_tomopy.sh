#/usr/bin/env bash
 _tomopy()
{
	local cur prev opts
	COMPREPLY=()
	cur="${COMP_WORDS[COMP_CWORD]}"
	prev="${COMP_WORDS[COMP_CWORD-1]}"
	if [[ ${prev} == "recon" ]] ; then
		opts="--help --rotation-axis --rotation-axis-auto --binning --blocked-views --dark-zero --file-format --file-name --file-type --nsino --nsino-per-chunk --pixel-size-auto --reverse --scintillator-auto --start-row --dx-update --missing-angles-end --zinger-level-white --zinger-removal-method --zinger-size --air --fix-nan-and-inf --fix-nan-and-inf-value --flat-correction-method --minus-log --normalization-cutoff --remove-stripe-method --fw-filter --fw-level --fw-pad --ti-alpha --ti-nblock --sf-size --energy --pixel-size --propagation-distance --retrieve-phase-alpha --retrieve-phase-alpha-try\n --retrieve-phase-method --beam-hardening-method --center-row --filter-1-material --filter-1-thickness --filter-2-material --filter-2-thickness --filter-3-material --filter-3-thickness --sample-material --scintillator-material --reconstruction-mask\n --reconstruction-mask-ratio --reconstruction-type --gridrec-filter --lprec-fbp-filter --astrasart-bootstrap\n --astrasart-max-constraint --astrasart-method --astrasart-min-constraint --astrasart-num_iter --astrasart-proj-type --astrasirt-bootstrap\n --astrasirt-max-constraint --astrasirt-method --astrasirt-min-constraint --astrasirt-num_iter --astrasirt-proj-type --astracgls-bootstrap\n --astracgls-method --astracgls-num_iter --astracgls-proj-type --config"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--help" ]] ; then
		opts="10.0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--rotation-axis" ]] ; then
		opts="-1.0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--rotation-axis-auto" ]] ; then
		opts="-1.0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--binning" ]] ; then
		opts="0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--blocked-views" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--dark-zero" ]] ; then
		opts="-1"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--file-format" ]] ; then
		opts="dx"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--file-name" ]] ; then
		opts="."
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--file-type" ]] ; then
		opts="standard"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--nsino" ]] ; then
		opts="0.5"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--nsino-per-chunk" ]] ; then
		opts="32"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--pixel-size-auto" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--reverse" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--scintillator-auto" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--start-row" ]] ; then
		opts="0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--dx-update" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--missing-angles-end" ]] ; then
		opts="800.0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--zinger-level-white" ]] ; then
		opts="1000.0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--zinger-removal-method" ]] ; then
		opts="none"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--zinger-size" ]] ; then
		opts="3"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--air" ]] ; then
		opts="10"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--fix-nan-and-inf" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--fix-nan-and-inf-value" ]] ; then
		opts="0.0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--flat-correction-method" ]] ; then
		opts="standard"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--minus-log" ]] ; then
		opts="True"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--normalization-cutoff" ]] ; then
		opts="1.0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--remove-stripe-method" ]] ; then
		opts="none"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--fw-filter" ]] ; then
		opts="sym16"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--fw-level" ]] ; then
		opts="7"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--fw-pad" ]] ; then
		opts="1"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--ti-alpha" ]] ; then
		opts="1.5"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--ti-nblock" ]] ; then
		opts="0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--sf-size" ]] ; then
		opts="5"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--energy" ]] ; then
		opts="20"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--pixel-size" ]] ; then
		opts="1.17"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--propagation-distance" ]] ; then
		opts="60"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--retrieve-phase-alpha" ]] ; then
		opts="0.001"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--retrieve-phase-alpha-try\n" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--retrieve-phase-method" ]] ; then
		opts="none"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--beam-hardening-method" ]] ; then
		opts="none"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--center-row" ]] ; then
		opts="0.0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--filter-1-material" ]] ; then
		opts="none"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--filter-1-thickness" ]] ; then
		opts="0.0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--filter-2-material" ]] ; then
		opts="none"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--filter-2-thickness" ]] ; then
		opts="0.0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--filter-3-material" ]] ; then
		opts="Be"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--filter-3-thickness" ]] ; then
		opts="750.0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--sample-material" ]] ; then
		opts="Fe"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--scintillator-material" ]] ; then
		opts="gridrec"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--reconstruction-mask\n" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--reconstruction-mask-ratio" ]] ; then
		opts="1.0"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--reconstruction-type" ]] ; then
		opts="try"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--gridrec-filter" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--lprec-fbp-filter" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astrasart-bootstrap\n" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astrasart-max-constraint" ]] ; then
		opts="None"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astrasart-method" ]] ; then
		opts="SART_CUDA"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astrasart-min-constraint" ]] ; then
		opts="None"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astrasart-num_iter" ]] ; then
		opts="200"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astrasart-proj-type" ]] ; then
		opts="cuda"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astrasirt-bootstrap\n" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astrasirt-max-constraint" ]] ; then
		opts="None"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astrasirt-method" ]] ; then
		opts="SIRT_CUDA"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astrasirt-min-constraint" ]] ; then
		opts="None"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astrasirt-num_iter" ]] ; then
		opts="200"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astrasirt-proj-type" ]] ; then
		opts="cuda"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astracgls-bootstrap\n" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astracgls-method" ]] ; then
		opts="CGLS_CUDA"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astracgls-num_iter" ]] ; then
		opts="200"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--astracgls-proj-type" ]] ; then
		opts="cuda"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
	if [[ ${prev} == "--config" ]] ; then
		opts="False"
		COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
		return 0
	fi
}
complete -F _tomopy tomopy