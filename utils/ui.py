"""Presentation helpers for the Divot road-intelligence interface."""

from __future__ import annotations

import html

import streamlit as st

# Streamlit sanitises st.html and removes <script>, so anything that needs real
# JavaScript has to go through st.iframe, which renders in an iframe. Passing
# markup rather than a URL is supported: a src that matches no URL pattern is
# embedded as raw HTML.
_SCENE_COLORS = {
    "ground": "#080e15",
    "dot": "#3c5266",
    "measure": "#45cdff",
    "detect": "#ffcb1f",
    "ink": "#eaf2f8",
    "dim": "#71889c",
}


def apply_app_style() -> None:
    """Apply the dark instrument-panel visual system."""

    st.html("""
        <style>
        :root {
            --ground: #0d141c;
            --ground-deep: #080e15;
            --panel: #151f2a;
            --panel-high: #1e2b38;
            --line: #2d4052;
            --line-soft: #223140;
            --ink: #eaf2f8;
            --ink-soft: #9db2c4;
            --ink-dim: #71889c;
            --detect: #ffcb1f;
            --detect-soft: rgba(255, 203, 31, .17);
            --measure: #45cdff;
            --measure-soft: rgba(69, 205, 255, .19);
            --miss: #ff6b42;
            --miss-soft: rgba(255, 107, 66, .17);
            --display: "IBM Plex Serif", Georgia, serif;
            --body: "Inter", system-ui, sans-serif;
            --mono: "JetBrains Mono", monospace;
            --ease: cubic-bezier(.2, .8, .25, 1);
        }

        html { scroll-behavior: smooth; }
        html, body, .stApp { overflow-x: clip; }
        .stApp {
            color: var(--ink);
            background:
                radial-gradient(circle at 78% -4%, var(--measure-soft), transparent 30rem),
                repeating-linear-gradient(0deg, transparent 0 3px, rgba(234,242,248,.022) 3px 4px),
                var(--ground);
        }
        .block-container { max-width: 1420px; padding-top: 1.5rem; padding-bottom: 6rem; }
        .stMainBlockContainer { min-width: 0; }
        [data-testid="stHeader"] { background: linear-gradient(180deg, rgba(13,20,28,.96), rgba(13,20,28,0)); }
        [data-testid="stHeaderActionElements"] { color: var(--ink-soft); }


        /* nav rows behave like real controls: numbered, tracked, and they
           reveal an arrow on approach */
        [data-testid="stSidebarNav"] ul { counter-reset: navrow; }
        [data-testid="stSidebarNav"] a { counter-increment: navrow; overflow: hidden; }
        [data-testid="stSidebarNav"] a::after {
            content: "0" counter(navrow);
            position: absolute; right: .85rem; top: 50%; transform: translateY(-50%);
            font-family: var(--mono); font-size: .55rem; letter-spacing: .1em;
            color: var(--ink-dim); opacity: .55; transition: opacity 200ms var(--ease), color 200ms var(--ease);
        }
        [data-testid="stSidebarNav"] a:hover::after { opacity: 1; color: var(--detect); }
        [data-testid="stSidebarNav"] a[aria-current="page"]::after { opacity: 1; color: var(--detect); }
        [data-testid="stSidebarNav"] a span[data-testid="stIconMaterial"] {
            transition: transform 260ms cubic-bezier(.22,1.2,.36,1), color 200ms var(--ease);
        }
        [data-testid="stSidebarNav"] a:hover span[data-testid="stIconMaterial"] { transform: scale(1.18) rotate(-6deg); }
        [data-testid="stSidebarNav"] a[aria-current="page"] span[data-testid="stIconMaterial"] { color: var(--detect); }

        /* the live strip under the mark */
        [data-testid="stSidebar"] iframe { display: block; width: 100%; border: 0; border-radius: 2px; }
        .sidebar-scope {
            display: flex; align-items: center; justify-content: space-between; gap: .5rem;
            margin: 1.1rem 0 .45rem; color: var(--ink-dim);
            font-family: var(--mono); font-size: .5rem; letter-spacing: .16em; text-transform: uppercase;
        }
        .sidebar-scope i {
            width: 5px; height: 5px; border-radius: 50%; background: var(--measure); flex: none;
            animation: sc-pulse 1.8s ease-in-out infinite;
        }
        .sidebar-scope span { flex: 1; height: 1px; background: var(--line-soft); }

        /* ------------------------------------------------------- sign-in */
        .auth-head { max-width: 30rem; margin: 0 auto 1.6rem; text-align: left; }
        .auth-mark {
            width: 44px; height: 44px; margin-bottom: 1.4rem;
            background: url('/app/static/divot-mark.svg') center / contain no-repeat;
        }
        .auth-head h1 {
            margin: 0; color: var(--ink); font-family: var(--display);
            font-size: clamp(1.9rem, 3vw, 2.5rem); font-weight: 600;
            line-height: 1.1; letter-spacing: -.03em;
        }
        .auth-head p {
            margin: .8rem 0 0; color: var(--ink-soft); font-size: .95rem; line-height: 1.6;
        }
        .st-key-auth_panel {
            max-width: 30rem; margin: 0 auto; padding: 1.7rem;
            border: 1px solid var(--line); border-radius: 2px; background: var(--panel);
        }
        .account-badge {
            display: flex; align-items: center; gap: .5rem;
            margin: .9rem 0 .2rem; padding: .5rem .6rem;
            border: 1px solid var(--line-soft); border-radius: 2px; background: var(--panel);
            font-family: var(--mono); font-size: .55rem; letter-spacing: .12em;
            text-transform: uppercase; color: var(--ink-soft);
        }
        .account-badge__dot {
            width: 5px; height: 5px; border-radius: 50%; background: var(--detect); flex: none;
        }
        .account-badge__name { color: var(--ink); }

        /* ------------------------------------------------- sidebar (base) */
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--line-soft);
            background: linear-gradient(180deg, rgba(69,205,255,.10), transparent 30%), var(--ground-deep);
        }
        [data-testid="stSidebarNav"] { padding-top: .2rem; }
        [data-testid="stSidebarNav"] a {
            position: relative; min-height: 2.9rem; margin-bottom: .2rem; padding-left: 1rem;
            border: 1px solid transparent; border-radius: 2px;
            color: var(--ink-soft); white-space: nowrap;
            transition: color 200ms var(--ease), background 200ms var(--ease), padding-left 200ms var(--ease);
        }
        [data-testid="stSidebarNav"] a::before {
            content: ""; position: absolute; left: 0; top: 50%; width: 2px; height: 0;
            background: var(--detect); transform: translateY(-50%);
            transition: height 220ms var(--ease);
        }
        [data-testid="stSidebarNav"] a:hover {
            color: var(--ink); background: rgba(234,242,248,.055); padding-left: 1.25rem;
        }
        [data-testid="stSidebarNav"] a:hover::before { height: 40%; }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            color: var(--ink); background: var(--panel); padding-left: 1.25rem;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"]::before { height: 100%; }

        .surface-brand { padding: 1.35rem .2rem 1.45rem; border-bottom: 1px solid var(--line-soft); }
        .surface-brand__row { display: flex; align-items: center; gap: .8rem; }
        .surface-brand__mark {
            display: block; width: 44px; height: 44px; flex: none;
            background: url('/app/static/divot-mark.svg') center / contain no-repeat;
            transition: transform 320ms cubic-bezier(.22,1.2,.36,1);
        }
        .surface-brand:hover .surface-brand__mark { transform: translateY(-2px) rotate(-3deg); }
        .surface-brand__text { display: flex; flex-direction: column; gap: .28rem; min-width: 0; }
        .surface-brand__name {
            color: var(--ink); font-family: var(--body); font-size: 1.02rem; font-weight: 700;
            letter-spacing: .01em; line-height: 1;
        }
        .surface-brand__name span { color: var(--detect); }
        .surface-brand__meta {
            color: var(--ink-dim); font-family: var(--mono);
            font-size: .5rem; letter-spacing: .17em; text-transform: uppercase; line-height: 1.2;
        }

        /* ------------------------------------------------------------- hero */
        .st-key-hero_shell {
            position: relative; overflow: hidden; margin: .5rem 0 0;
            border: 1px solid var(--line); border-radius: 2px; background: var(--panel);
        }
        .st-key-hero_shell [data-testid="stColumn"]:last-child {
            position: relative; align-self: stretch;
            border-left: 1px solid var(--line); background: var(--ground-deep);
        }
        .st-key-hero_shell [data-testid="stColumn"]:last-child [data-testid="stElementContainer"] {
            margin: 0; position: static;
        }
        .st-key-hero_shell iframe { display: block; width: 100%; border: 0; }
        .scene-tag {
            position: absolute; z-index: 4; left: 0; right: 0; bottom: 0; pointer-events: none;
            display: flex; align-items: center; gap: .6rem;
            padding: .55rem .9rem; border-top: 1px solid var(--line-soft);
            color: var(--ink-dim); background: linear-gradient(180deg, rgba(8,14,21,0), rgba(8,14,21,.92) 55%);
            font-family: var(--mono); font-size: .5rem; letter-spacing: .16em;
        }
        .scene-tag::before {
            content: ""; width: 5px; height: 5px; border-radius: 50%; background: var(--measure);
            flex: none; animation: sc-pulse 1.8s ease-in-out infinite;
        }
        .surface-hero__copy {
            position: relative; z-index: 3; display: flex; flex-direction: column; justify-content: center;
            min-width: 0; padding: clamp(2rem, 3.6vw, 3.4rem);
        }
        .surface-kicker {
            display: flex; align-items: center; gap: .75rem; margin-bottom: 1.5rem;
            color: var(--ink-dim); font-family: var(--mono); font-size: .62rem;
            font-weight: 500; letter-spacing: .16em; text-transform: uppercase;
            animation: sc-rise 500ms var(--ease) both;
        }
        .surface-kicker::before { content: ""; width: 22px; height: 1px; background: var(--detect); flex: none; }
        .surface-hero__copy h1 {
            max-width: 14ch; margin: 0; color: var(--ink); font-family: var(--display);
            font-size: clamp(2.3rem, 4.4vw, 4.2rem); font-weight: 600; line-height: 1.04;
            letter-spacing: -.035em; text-wrap: balance; hyphens: none; min-width: 0;
            animation: sc-rise 620ms var(--ease) 60ms both;
        }
        .surface-hero__description {
            max-width: 46ch; margin: 1.5rem 0 0; padding-left: 1.1rem;
            border-left: 1px solid var(--line); color: var(--ink-soft);
            font-size: 1.01rem; line-height: 1.68;
            animation: sc-rise 620ms var(--ease) 140ms both;
        }
        .surface-status {
            display: inline-flex; align-items: center; align-self: flex-start; gap: .55rem;
            margin-top: 1.8rem; padding: .5rem .75rem; border: 1px solid var(--line);
            border-radius: 2px; color: var(--ink-soft); background: var(--ground-deep);
            font-family: var(--mono); font-size: .58rem; font-weight: 500;
            letter-spacing: .11em; text-transform: uppercase;
            animation: sc-rise 620ms var(--ease) 220ms both;
        }
        .surface-status i {
            width: 6px; height: 6px; border-radius: 50%; background: var(--miss); flex: none;
        }


        /* ------------------------------------------------------------ brief */
        .surface-brief {
            margin: 2.6rem 0 5rem; border: 1px solid var(--line-soft);
            border-left: 2px solid var(--detect); border-radius: 2px;
            background: var(--panel); overflow: hidden;
        }
        .surface-brief__statement { padding: clamp(2rem, 4.2vw, 3.6rem); background: var(--panel); }
        .surface-brief__statement strong {
            display: block; max-width: 26ch; color: var(--ink); font-family: var(--display);
            font-size: clamp(1.8rem, 3.1vw, 3rem); font-weight: 500; line-height: 1.12; letter-spacing: -.03em;
        }

        /* ------------------------------------------------------- page intro */
        .surface-intro {
            position: relative; overflow: hidden; margin: .5rem 0 3rem;
            padding: clamp(2.4rem, 5vw, 4rem); border: 1px solid var(--line);
            border-radius: 2px; background: var(--panel);
        }
        .surface-intro::before {
            content: ""; position: absolute; inset: 0; pointer-events: none;
            background: radial-gradient(680px 220px at 88% -30%, var(--measure-soft), transparent 70%);
        }
        .surface-intro::after {
            content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 2px;
            background: var(--detect); animation: sc-draw-y 620ms var(--ease) both;
        }
        .surface-intro__index {
            position: relative; color: var(--ink-dim); font-family: var(--mono); font-size: .6rem;
            font-weight: 500; letter-spacing: .16em; text-transform: uppercase;
            animation: sc-rise 480ms var(--ease) both;
        }
        .surface-intro h1 {
            position: relative; max-width: 20ch; margin: .9rem 0 .9rem; color: var(--ink);
            font-family: var(--display); font-size: clamp(2.5rem, 4.4vw, 4.2rem); font-weight: 600;
            line-height: 1.04; letter-spacing: -.035em; text-wrap: balance; min-width: 0;
            animation: sc-rise 560ms var(--ease) 60ms both;
        }
        .surface-intro p {
            position: relative; max-width: 64ch; margin: 0; color: var(--ink-soft); line-height: 1.68;
            animation: sc-rise 560ms var(--ease) 130ms both;
        }

        /* ---------------------------------------------------- section rules */
        .surface-section {
            position: relative; display: grid; grid-template-columns: minmax(0, 1fr) minmax(260px, .55fr);
            gap: 2rem; align-items: end; margin: 4.2rem 0 1.4rem; padding: 0 0 1.1rem;
            border-bottom: 1px solid var(--line-soft);
        }
        .surface-section::after {
            content: ""; position: absolute; left: 0; bottom: -1px; height: 1px;
            width: 100%; background: var(--detect); animation: sc-draw-x 900ms var(--ease) both;
        }
        .surface-section strong {
            color: var(--ink); font-family: var(--display); font-size: clamp(1.7rem, 2.7vw, 2.5rem);
            font-weight: 500; line-height: 1.14; letter-spacing: -.03em;
        }
        .surface-section span { max-width: 54ch; color: var(--ink-soft); font-size: .88rem; line-height: 1.6; }

        /* ------------------------------------------------------ action tiles */
        .st-key-image_action, .st-key-video_action {
            position: relative; min-height: 390px; overflow: hidden; padding: 2rem;
            border: 1px solid var(--line); border-radius: 2px; background: var(--panel); cursor: pointer;
            transition: border-color 220ms var(--ease), background 220ms var(--ease), transform 220ms var(--ease);
        }
        .st-key-image_action::before, .st-key-video_action::before {
            content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px;
            background: var(--measure); transform: scaleX(0); transform-origin: left;
            transition: transform 300ms var(--ease);
        }
        .st-key-video_action::before { background: var(--detect); }
        .st-key-image_action::after, .st-key-video_action::after {
            content: ""; position: absolute; inset: 0; pointer-events: none;
            background: radial-gradient(420px 200px at 82% 108%, var(--measure-soft), transparent 72%);
            opacity: 0; transition: opacity 260ms var(--ease);
        }
        .st-key-video_action::after { background: radial-gradient(420px 200px at 82% 108%, var(--detect-soft), transparent 72%); }
        .st-key-image_action:hover, .st-key-video_action:hover,
        .st-key-image_action:focus-within, .st-key-video_action:focus-within {
            background: var(--panel-high); border-color: var(--line); transform: translateY(-3px);
        }
        .st-key-image_action:hover::before, .st-key-video_action:hover::before,
        .st-key-image_action:focus-within::before, .st-key-video_action:focus-within::before { transform: scaleX(1); }
        .st-key-image_action:hover::after, .st-key-video_action:hover::after,
        .st-key-image_action:focus-within::after, .st-key-video_action:focus-within::after { opacity: 1; }
        .action-type {
            display: inline-flex; padding: .34rem .5rem; border: 1px solid var(--line); border-radius: 2px;
            color: var(--ink-dim); font-family: var(--mono); font-size: .55rem; font-weight: 500;
            letter-spacing: .13em; text-transform: uppercase;
        }
        .st-key-image_action h3, .st-key-video_action h3 {
            position: relative; z-index: 1; max-width: 14ch; margin-top: 4.6rem; color: var(--ink);
            font-family: var(--display); font-size: clamp(2rem, 3.1vw, 3rem); font-weight: 500;
            line-height: 1.08; letter-spacing: -.03em;
        }
        .st-key-image_action p, .st-key-video_action p {
            position: relative; z-index: 1; max-width: 42ch; color: var(--ink-soft); font-size: .92rem;
        }
        .st-key-image_action > [data-testid="stElementContainer"]:has(> [data-testid="stPageLink"]),
        .st-key-video_action > [data-testid="stElementContainer"]:has(> [data-testid="stPageLink"]) {
            position: absolute; z-index: 10; inset: 0; width: 100%; height: 100%; margin: 0;
        }
        .st-key-image_action [data-testid="stPageLink"], .st-key-video_action [data-testid="stPageLink"],
        .st-key-image_action [data-testid="stPageLink"] > div, .st-key-video_action [data-testid="stPageLink"] > div {
            width: 100%; height: 100%; margin: 0;
        }
        .st-key-image_action [data-testid="stPageLink"] a, .st-key-video_action [data-testid="stPageLink"] a {
            position: absolute; inset: 0; width: 100%; height: 100%; padding: 0;
            border: 0; opacity: 0; box-shadow: none;
        }
        .st-key-image_action [data-testid="stPageLink"] a:hover, .st-key-video_action [data-testid="stPageLink"] a:hover {
            transform: none; box-shadow: none;
        }

        /* ------------------------------------------------------- run status */
        .surface-media-stamp { display: flex; flex-wrap: wrap; gap: .4rem; margin: .7rem 0 1.1rem; }
        .surface-media-stamp span {
            padding: .36rem .55rem; border: 1px solid var(--line-soft); border-radius: 2px;
            color: var(--ink-soft); background: var(--panel); font-family: var(--mono);
            font-size: .53rem; letter-spacing: .09em; text-transform: uppercase;
        }
        .surface-media-stamp span:first-child {
            color: var(--detect); border-color: rgba(255,203,31,.42); background: var(--detect-soft);
        }
        .scan-banner {
            position: relative; overflow: hidden; margin: .8rem 0; padding: .95rem 1rem .95rem 3rem;
            border: 1px solid var(--line); border-radius: 2px; color: var(--ink-soft);
            background: var(--panel); font-family: var(--mono); font-size: .63rem; letter-spacing: .06em;
        }
        .scan-banner::before {
            content: ""; position: absolute; left: 1.15rem; top: 50%; width: 7px; height: 7px;
            margin-top: -3.5px; border-radius: 50%; background: var(--measure);
            animation: sc-pulse 1.3s ease-in-out infinite;
        }
        .scan-banner::after {
            content: ""; position: absolute; left: 0; bottom: 0; width: 34%; height: 2px;
            background: linear-gradient(90deg, transparent, var(--measure), transparent);
            animation: sc-indeterminate 1.5s ease-in-out infinite;
        }
        .result-arrive {
            height: 2px; margin: 1.6rem 0 0;
            background: linear-gradient(90deg, var(--detect), var(--measure) 46%, transparent);
            animation: sc-draw-x 620ms var(--ease) both;
        }

        /* ----------------------------------------------------- widget skins */
        div[data-testid="stMetric"] {
            position: relative; overflow: hidden; min-height: 128px; padding: 1.05rem 1.15rem;
            border: 1px solid var(--line-soft); border-radius: 2px; background: var(--panel);
            transition: border-color 200ms var(--ease), background 200ms var(--ease);
        }
        div[data-testid="stMetric"]:hover { border-color: var(--line); background: var(--panel-high); }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--ink-dim); font-family: var(--mono); font-size: .57rem; font-weight: 500;
            letter-spacing: .13em; text-transform: uppercase;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--ink); font-family: var(--display); font-weight: 600;
            letter-spacing: -.03em; font-variant-numeric: tabular-nums;
            animation: sc-rise 460ms var(--ease) both;
        }
        div[data-testid="stMetric"]::after {
            content: ""; position: absolute; left: 0; bottom: 0; height: 2px; width: 34%;
            background: var(--detect); animation: sc-draw-x 780ms var(--ease) both;
        }

        [data-testid="stFileUploaderDropzone"] {
            min-height: 156px; border: 1px dashed var(--line); border-radius: 2px;
            background: var(--panel);
            transition: border-color 200ms var(--ease), background 200ms var(--ease);
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--measure); background: var(--measure-soft);
        }
        [data-testid="stForm"] { border: 1px solid var(--line-soft); border-radius: 2px; background: var(--panel); }
        [data-testid="stImage"] img, [data-testid="stVideo"] video {
            border: 1px solid var(--line); border-radius: 2px;
        }
        .stButton button, .stDownloadButton button, [data-testid="stPageLink"] a {
            border-radius: 2px; font-weight: 600; cursor: pointer; white-space: nowrap;
            transition: transform 160ms var(--ease), border-color 160ms var(--ease), background 160ms var(--ease);
        }
        .stButton button:hover, .stDownloadButton button:hover, [data-testid="stPageLink"] a:hover {
            transform: translateY(-2px);
        }
        .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
            color: var(--ground-deep); background: var(--detect); border-color: var(--detect); font-weight: 700;
        }
        button:focus-visible, a:focus-visible, [role="tab"]:focus-visible,
        input:focus-visible, [data-testid="stFileUploaderDropzone"]:focus-within {
            outline: 2px solid var(--measure) !important; outline-offset: 3px;
        }
        [data-baseweb="tab-list"] { gap: .3rem; border-bottom: 1px solid var(--line-soft); }
        [data-baseweb="tab"] {
            border-radius: 2px 2px 0 0; color: var(--ink-dim); font-family: var(--mono);
            font-size: .6rem; font-weight: 500; letter-spacing: .1em; text-transform: uppercase;
        }
        [data-baseweb="tab"][aria-selected="true"] { color: var(--detect); }
        [data-testid="stDataFrame"] { border: 1px solid var(--line-soft); border-radius: 2px; }
        code, [data-testid="stCode"] { font-variant-ligatures: none; }

        /* -------------------------------------------------------- keyframes */
        @keyframes sc-rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
        @keyframes sc-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .22; } }
        @keyframes sc-pan { from { transform: scale(1.04) translate3d(0, 0, 0); } to { transform: scale(1.1) translate3d(-1.5%, -1%, 0); } }
        @keyframes sc-sweep {
            0%, 6% { top: -8%; opacity: 0; }
            14% { opacity: 1; }
            84% { opacity: 1; }
            96%, 100% { top: 100%; opacity: 0; }
        }
        @keyframes sc-lock {
            0%, 48% { opacity: 0; transform: scale(1.06); }
            56%, 92% { opacity: 1; transform: scale(1); }
            100% { opacity: 0; transform: scale(1); }
        }
        @keyframes sc-draw-x { from { transform: scaleX(0); transform-origin: left; } to { transform: scaleX(1); transform-origin: left; } }
        @keyframes sc-draw-y { from { transform: scaleY(0); transform-origin: top; } to { transform: scaleY(1); transform-origin: top; } }
        @keyframes sc-indeterminate {
            0% { left: -34%; } 100% { left: 100%; }
        }
        @keyframes sc-unmask {
            from { clip-path: inset(0 0 104% 0); transform: translateY(12px); }
            to { clip-path: inset(0 0 -12% 0); transform: none; }
        }
        @keyframes sc-shine {
            0%, 58% { transform: translateX(-110%); }
            100% { transform: translateX(110%); }
        }
        @keyframes sc-grid { to { background-position: 0 96px; } }

        /* ==================================================================
           MODERN MOTION LAYER
           Scroll-driven where the browser supports it, so entrance motion is
           tied to where the reader is rather than replaying on every
           Streamlit rerun. Everything below degrades to the static styles
           above when unsupported.

           No @property is used here: Streamlit sanitizes st.html and drops
           the whole stylesheet when it encounters that at-rule. For the same
           reason no comment in this file may contain a literal HTML tag --
           the parser reads it as real markup and discards everything after.
           ================================================================== */

        /* a readout sweep across the two states worth noticing */
        .stButton button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            position: relative; overflow: hidden;
        }
        .stButton button[kind="primary"]::after,
        [data-testid="stFormSubmitButton"] button[kind="primary"]::after {
            content: ""; position: absolute; inset: 0; pointer-events: none;
            background: linear-gradient(105deg, transparent 34%,
                rgba(255,255,255,.30) 50%, transparent 66%);
            transform: translateX(-110%);
        }
        .stButton button[kind="primary"]:hover::after,
        [data-testid="stFormSubmitButton"] button[kind="primary"]:hover::after {
            animation: sc-shine 900ms ease-out;
        }

        /* hero headline reveals as a line of set type, not a faded block */
        .surface-hero__copy h1 { animation: sc-unmask 760ms var(--ease) 60ms both; }
        .surface-intro h1 { animation: sc-unmask 680ms var(--ease) 60ms both; }

        /* survey grid drifting across the road plate */

        /* action tiles gain depth on approach */
        .st-key-image_action, .st-key-video_action {
            transform: perspective(1400px); transform-style: preserve-3d;
            transition: transform 320ms cubic-bezier(.22,1.2,.36,1),
                        border-color 220ms var(--ease), background 220ms var(--ease);
        }
        .st-key-image_action:hover, .st-key-video_action:hover,
        .st-key-image_action:focus-within, .st-key-video_action:focus-within {
            transform: perspective(1400px) rotateX(2.2deg) translateY(-6px) scale(1.008);
        }
        .st-key-image_action h3, .st-key-video_action h3,
        .st-key-image_action .action-type, .st-key-video_action .action-type {
            transition: transform 340ms cubic-bezier(.22,1.2,.36,1);
        }
        .st-key-image_action:hover h3, .st-key-video_action:hover h3 { transform: translateZ(28px); }
        .st-key-image_action:hover .action-type, .st-key-video_action:hover .action-type { transform: translateZ(16px); }

        @supports (animation-timeline: view()) {
            /* entrance is now a function of scroll position, not of time */
            .surface-brief, .surface-section, .surface-media-stamp,
            .st-key-image_action, .st-key-video_action, div[data-testid="stMetric"] {
                animation: sc-rise linear both;
                animation-timeline: view();
                animation-range: entry 4% cover 26%;
            }
            .surface-section::after, div[data-testid="stMetric"]::after {
                animation: sc-draw-x linear both;
                animation-timeline: view();
                animation-range: entry 12% cover 34%;
            }
            /* the tiles keep their spring hover on top of the scroll reveal */
            .st-key-image_action, .st-key-video_action { animation-composition: add; }
        }

        /* Reading progress. The bar is fixed, which takes it out of the
           scroll-container chain, so scroll(nearest) would resolve to an
           inactive timeline. Naming the timeline on the real scroll port and
           publishing it through the app root is what makes it read. */
        @supports (timeline-scope: --sc-main) {
            .stApp { timeline-scope: --sc-main; }
            [data-testid="stMain"] { scroll-timeline: --sc-main block; }
            .stApp::before {
                content: ""; position: fixed; z-index: 999; left: 0; top: 0;
                width: 100%; height: 2px; transform-origin: 0 50%;
                background: linear-gradient(90deg, var(--detect), var(--measure));
                animation: sc-draw-x linear both;
                animation-timeline: --sc-main;
            }
        }

        /* ------------------------------------------------------- responsive */
        @media (max-width: 1100px) {
            .surface-section { grid-template-columns: minmax(0, 1fr); gap: .7rem; }
            /* the hero keeps two columns down to Streamlit own stacking point,
               so the headline has to give way before the column gets cramped */
            .surface-hero__copy { padding: clamp(1.5rem, 3vw, 2.4rem); }
            .surface-hero__copy h1 { font-size: clamp(1.85rem, 4.6vw, 2.7rem); }
            .surface-hero__description { margin-top: 1.15rem; font-size: .95rem; }
        }
        @media (max-width: 768px) {
            .block-container { padding: .7rem 1rem 4rem; }
            .surface-hero__copy { padding: 2rem 1.45rem; }
            .surface-hero__copy h1 { font-size: clamp(2rem, 7vw, 2.7rem); }
            .surface-brief { margin: 2rem 0 3.6rem; }
            .surface-intro { padding: 2.4rem 1.5rem; }
            .surface-intro h1 { font-size: clamp(2.2rem, 9vw, 3rem); }
            .st-key-image_action, .st-key-video_action { min-height: 330px; }
            .st-key-image_action h3, .st-key-video_action h3 { margin-top: 3rem; }
        }
        @media (max-width: 480px) {
            .surface-hero__copy h1 { font-size: clamp(1.85rem, 8.5vw, 2.35rem); }
            .surface-hero__description { font-size: .94rem; }
            .surface-brief__statement strong { font-size: 1.7rem; }
            .surface-section { margin-top: 3rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: .01ms !important;
                animation-iteration-count: 1 !important;
                animation-timeline: auto !important;
                transition: none !important;
            }
            .surface-hero__copy h1, .surface-intro h1 { clip-path: none; }
            .stButton button[kind="primary"]::after,
            [data-testid="stFormSubmitButton"] button[kind="primary"]::after { display: none; }
            .st-key-image_action:hover, .st-key-video_action:hover,
            .st-key-image_action:focus-within, .st-key-video_action:focus-within { transform: none; }
            .stApp::before { display: none; }
        }
        </style>
        """)


def _scene_markup(height: int, compact: bool = False) -> str:
    """Build the point-cloud road scene rendered inside a component iframe."""

    colors = _SCENE_COLORS
    mode = "1" if compact else "0"
    return f"""
<style>
  html, body {{ margin: 0; padding: 0; background: {colors["ground"]}; overflow: hidden; }}
  canvas {{ display: block; width: 100%; height: {height}px; cursor: crosshair; }}
</style>
<canvas id="scene"></canvas>
<script>
(function () {{
  "use strict";
  var COMPACT = {mode} === 1;
  var C = {{
    dot: "{colors["dot"]}",
    measure: "{colors["measure"]}",
    detect: "{colors["detect"]}",
    ink: "{colors["ink"]}",
    dim: "{colors["dim"]}",
    ground: "{colors["ground"]}"
  }};
  var reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  var cv = document.getElementById("scene");
  var g = cv.getContext("2d");
  var W = 0, H = 0, FOV = 300, dpr = Math.min(devicePixelRatio || 1, 2);

  function resize() {{
    W = Math.max(cv.clientWidth, 1); H = {height};
    cv.width = W * dpr; cv.height = H * dpr;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    FOV = Math.max(70, W * (COMPACT ? 0.62 : 0.82));
  }}
  addEventListener("resize", resize); resize();

  // --- the road surface, as a grid of points on a plane -------------------
  var STEP = COMPACT ? 0.12 : 0.075;
  var XR = COMPACT ? 2.2 : 3.5, Z0 = 0.35, Z1 = COMPACT ? 5.0 : 7.6;
  // rows are spaced so they land evenly on screen rather than evenly in the
  // world, otherwise everything piles up at the horizon and the near field
  // reads as empty
  var ROWS = COMPACT ? 34 : 54;
  var invN = 1 / (Z0 + 0.5), invF = 1 / (Z1 + 0.5);
  var pts = [];
  for (var x = -XR; x <= XR; x += STEP) {{
    for (var r = 0; r < ROWS; r++) {{
      var inv = invN + (invF - invN) * (r / (ROWS - 1));
      pts.push({{ x: x, z: 1 / inv - 0.5 }});
    }}
  }}

  // Depressions in that plane. The last one is never reported, which is what
  // a recall of 0.41 actually looks like.
  var holes = COMPACT
    ? [{{ x: -0.45, z: 1.45, r: 0.42, d: 0.26, conf: 0.88 }},
       {{ x:  0.55, z: 2.60, r: 0.38, d: 0.22, conf: 0.51 }}]
    : [{{ x: -0.55, z: 1.35, r: 0.55, d: 0.34, conf: 0.91 }},
       {{ x:  0.70, z: 2.15, r: 0.52, d: 0.32, conf: 0.62 }},
       {{ x: -0.62, z: 3.30, r: 0.55, d: 0.34, conf: 0.44 }},
       {{ x:  0.85, z: 4.70, r: 0.48, d: 0.24, conf: 0.00 }}];

  var CAM_Y = COMPACT ? 0.50 : 0.89, HORIZON = COMPACT ? 0.16 : 0.23;
  var tiltX = 0, tiltY = 0, tx = 0, ty = 0;
  var mouse = {{ x: -9999, y: -9999, on: false }};

  cv.addEventListener("pointermove", function (e) {{
    var r = cv.getBoundingClientRect();
    mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; mouse.on = true;
    tx = ((mouse.x / W) - 0.5) * 2;
    ty = ((mouse.y / H) - 0.5) * 2;
  }});
  cv.addEventListener("pointerleave", function () {{
    mouse.on = false; mouse.x = mouse.y = -9999; tx = 0; ty = 0;
  }});

  function surfaceY(x, z) {{
    var y = 0, w = 0;
    for (var i = 0; i < holes.length; i++) {{
      var p = holes[i], dx = x - p.x, dz = z - p.z;
      var d = Math.sqrt(dx * dx + dz * dz);
      if (d < p.r) {{
        var t = 1 - d / p.r, f = t * t * (3 - 2 * t);
        y += p.d * f; if (f > w) w = f;
      }}
    }}
    return [y, w];
  }}

  function project(x, y, z) {{
    var d = z + 0.5;
    var s = FOV / d;
    return [W / 2 + (x + tiltX * 0.30) * s,
            H * HORIZON + (CAM_Y + y + tiltY * 0.16) * s, s];
  }}

  var t = 0, scanZ = Z0, PERIOD = COMPACT ? 3.8 : 5.4;

  function frame() {{
    t += 1 / 60;
    tiltX += (tx - tiltX) * 0.06;
    tiltY += (ty - tiltY) * 0.06;
    scanZ = reduce ? Z1 : Z0 + ((t % PERIOD) / PERIOD) * (Z1 - Z0);

    g.fillStyle = C.ground; g.fillRect(0, 0, W, H);

    // horizon wash
    var hzEnd = H * HORIZON + 70;
    var hz = g.createLinearGradient(0, 0, 0, hzEnd);
    hz.addColorStop(0, "rgba(69,205,255,0)");
    hz.addColorStop(0.70, "rgba(69,205,255,.20)");
    hz.addColorStop(1, "rgba(69,205,255,0)");
    g.fillStyle = hz; g.fillRect(0, 0, W, hzEnd);

    for (var i = 0; i < pts.length; i++) {{
      var p = pts[i];
      var sy = surfaceY(p.x, p.z), y = sy[0], w = sy[1];
      var pr = project(p.x, y, p.z), sx = pr[0], syy = pr[1], s = pr[2];
      if (syy < -20 || syy > H + 20) continue;

      // the cursor pushes the surface away from itself
      if (mouse.on) {{
        var mdx = sx - mouse.x, mdy = syy - mouse.y;
        var md = Math.sqrt(mdx * mdx + mdy * mdy);
        var R = COMPACT ? 55 : 95;
        if (md < R && md > 0.01) {{
          var push = (1 - md / R) * (COMPACT ? 10 : 18);
          sx += (mdx / md) * push; syy += (mdy / md) * push;
        }}
      }}

      var near = 1 - Math.min(Math.abs(p.z - scanZ) / 0.40, 1);
      var size = Math.max(0.6, s * 0.0085);
      var a = Math.min(0.92, 0.15 + s * 0.0014 + w * 0.75 + near * 0.5);

      var col = C.dot;
      if (near > 0.25) col = C.measure;
      else if (w > 0.08) col = C.detect;

      g.globalAlpha = a;
      g.fillStyle = col;
      g.fillRect(sx - size / 2, syy - size / 2, size, size);
    }}
    g.globalAlpha = 1;

    // brackets lock on once the sweep has gone past a depression
    g.lineWidth = 1.4;
    g.font = '600 9px ui-monospace, "JetBrains Mono", monospace';
    for (var h = 0; h < holes.length; h++) {{
      var p2 = holes[h];
      if (p2.conf < 0.35) continue;
      var passed = scanZ - (p2.z + p2.r * 0.5);
      if (passed < 0) continue;
      var e = Math.min(passed / 0.5, 1);
      e = 1 - Math.pow(1 - e, 3);

      // Two opposite corners of the flat footprint would sit above the dots
      // and skew in perspective, so walk the rim and take the real screen
      // extent, including the sunken centre.
      var minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
      for (var q = 0; q < 18; q++) {{
        var ang = q / 18 * Math.PI * 2;
        var wx = p2.x + Math.cos(ang) * p2.r;
        var wz = p2.z + Math.sin(ang) * p2.r;
        var rp = project(wx, surfaceY(wx, wz)[0], wz);
        if (rp[0] < minX) minX = rp[0];
        if (rp[0] > maxX) maxX = rp[0];
        if (rp[1] < minY) minY = rp[1];
        if (rp[1] > maxY) maxY = rp[1];
      }}
      var deep = project(p2.x, p2.d, p2.z);
      if (deep[1] > maxY) maxY = deep[1];
      var pad = 3;
      var bx = minX - pad, bw = (maxX - minX) + pad * 2;
      var by = minY - pad, bh = Math.max((maxY - minY) + pad * 2, 12);

      g.globalAlpha = e;
      g.strokeStyle = C.detect;
      var arm = Math.min(9, bw * 0.28) * e;
      [[bx, by, 1, 1], [bx + bw, by, -1, 1], [bx, by + bh, 1, -1], [bx + bw, by + bh, -1, -1]]
        .forEach(function (c) {{
          g.beginPath();
          g.moveTo(c[0] + c[2] * arm, c[1]);
          g.lineTo(c[0], c[1]);
          g.lineTo(c[0], c[1] + c[3] * arm);
          g.stroke();
        }});
      if (e > 0.6 && !COMPACT) {{
        var lab = p2.conf.toFixed(2);
        var lw = g.measureText(lab).width + 7;
        g.fillStyle = C.detect;
        g.fillRect(bx, by - 12, lw, 12);
        g.fillStyle = C.ground;
        g.textBaseline = "middle"; g.textAlign = "left";
        g.fillText(lab, bx + 3.5, by - 5.5);
      }}
      g.globalAlpha = 1;
    }}

    if (!reduce) requestAnimationFrame(frame);
  }}
  frame();
}})();
</script>
"""


def scan_scene(height: int = 545) -> None:
    """Render the interactive road point-cloud used as the home hero visual."""

    st.iframe(_scene_markup(height), height=height)


def sidebar_scene(height: int = 132) -> None:
    """Render the compact live scan strip shown in the sidebar."""

    st.iframe(_scene_markup(height, compact=True), height=height)


def signin_shell(title: str, description: str) -> None:
    """Render the header above the sign-in form."""

    st.html(f"""
        <div class="auth-head">
          <div class="auth-mark"></div>
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(description)}</p>
        </div>
        """)


def account_badge(username: str) -> None:
    """Show which account the sidebar belongs to."""

    st.html(f"""
        <div class="account-badge">
          <span class="account-badge__dot"></span>
          <span class="account-badge__name">{html.escape(username)}</span>
        </div>
        """)


def brand() -> None:
    """Render the application mark in the sidebar."""

    st.html("""
        <div class="surface-brand">
          <div class="surface-brand__row">
            <span class="surface-brand__mark" aria-hidden="true"></span>
            <span class="surface-brand__text">
              <span class="surface-brand__name">DIVOT<span>.</span></span>
              <span class="surface-brand__meta">Road evidence studio</span>
            </span>
          </div>
        </div>
        """)
    st.html('<div class="sidebar-scope">live surface<span></span><i></i></div>')
    sidebar_scene()


def hero(kicker: str, title: str, description: str, ready: bool | None = None) -> None:
    """Render the split editorial road-inspection hero."""

    model_status = ""
    if ready is False:
        model_status = '<div class="surface-status"><i></i>Checkpoint missing</div>'

    with st.container(key="hero_shell", border=False):
        copy_col, scene_col = st.columns([1, 1], vertical_alignment="center")
        with copy_col:
            st.html(f"""
                <div class="surface-hero__copy">
                  <div class="surface-kicker">{html.escape(kicker)}</div>
                  <h1>{html.escape(title)}</h1>
                  <p class="surface-hero__description">{html.escape(description)}</p>
                  {model_status}
                </div>
                """)
        with scene_col:
            st.html('<div class="scene-tag">LIVE SURFACE / DRAG TO LOOK</div>')
            scan_scene()


def hero_brief() -> None:
    """Render one concise product statement under the home hero."""

    st.html("""
        <div class="surface-brief">
          <div class="surface-brief__statement">
            <strong>Road media becomes evidence you can inspect—not a verdict you have to trust blindly.</strong>
          </div>
        </div>
        """)


def page_intro(index: str, kicker: str, title: str, description: str) -> None:
    """Render an editorial operational header for tool pages."""

    st.html(f"""
        <section class="surface-intro">
          <div class="surface-intro__index">{html.escape(index)} / {html.escape(kicker)}</div>
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(description)}</p>
        </section>
        """)


def section_heading(title: str, description: str | None = None) -> None:
    """Render a direct section heading and supporting description."""

    detail = f"<span>{html.escape(description)}</span>" if description else ""
    st.html(f"""
        <div class="surface-section">
          <strong>{html.escape(title)}</strong>
          {detail}
        </div>
        """)


def media_stamp(*items: str) -> None:
    """Render compact media metadata."""

    values = "".join(f"<span>{html.escape(item)}</span>" for item in items if item)
    st.html(f'<div class="surface-media-stamp">{values}</div>')


def scanning_banner(message: str) -> None:
    """Render a purposeful animated scan indicator."""

    st.html(f'<div class="scan-banner">{html.escape(message)}</div>')


def result_reveal() -> None:
    """Add a subtle entrance marker before fresh results."""

    st.html('<div class="result-arrive"></div>')
