"""Presentation helpers for the Surface/01 road-intelligence interface."""

from __future__ import annotations

import html

import streamlit as st


def apply_app_style() -> None:
    """Apply the civic field-journal visual system."""

    st.html("""
        <style>
        :root {
            --paper: #f4f0e6;
            --paper-high: #fffdf7;
            --paper-low: #e7e0d2;
            --ink: #12263f;
            --ink-soft: #536174;
            --midnight: #0e223c;
            --cobalt: #1646a0;
            --signal: #e64d2e;
            --signal-dark: #b9301d;
            --lime: #c9f24b;
            --line: #cbc2b2;
            --display: "DM Serif Display", Georgia, serif;
            --body: "Manrope", sans-serif;
            --mono: "JetBrains Mono", monospace;
            --ease: cubic-bezier(.2, .8, .2, 1);
        }

        html { scroll-behavior: smooth; }
        html, body, .stApp { overflow-x: clip; }
        .stApp {
            color: var(--ink);
            background:
                radial-gradient(circle at 82% 4%, rgba(22,70,160,.08), transparent 26rem),
                linear-gradient(rgba(18,38,63,.035) 1px, transparent 1px),
                var(--paper);
            background-size: auto, 100% 72px, auto;
        }
        .block-container { max-width: 1420px; padding-top: 1.5rem; padding-bottom: 6rem; }
        .stMainBlockContainer { min-width: 0; }
        [data-testid="stHeader"] { background: linear-gradient(180deg, rgba(244,240,230,.98), rgba(244,240,230,0)); }
        [data-testid="stHeaderActionElements"] { color: var(--ink); }

        [data-testid="stSidebar"] {
            border-right: 0;
            background: linear-gradient(145deg, rgba(255,255,255,.035), transparent 42%), var(--midnight);
        }
        [data-testid="stSidebarNav"] { padding-top: .2rem; }
        [data-testid="stSidebarNav"] a {
            min-height: 3.1rem; margin-bottom: .35rem; padding-left: .8rem;
            border: 1px solid transparent; border-radius: 2px;
            color: rgba(255,255,255,.72); white-space: nowrap;
            transition: color 180ms var(--ease), background 180ms var(--ease), transform 180ms var(--ease);
        }
        [data-testid="stSidebarNav"] a:hover {
            color: #fff; background: rgba(255,255,255,.07); transform: translateX(3px);
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            color: #fff; background: var(--signal); box-shadow: 5px 5px 0 rgba(201,242,75,.22);
        }

        .surface-brand { padding: 1.3rem .2rem 1.5rem; color: white; }
        .surface-brand__row { display: flex; align-items: center; gap: .8rem; }
        .surface-brand__mark {
            position: relative; display: grid; place-items: center; width: 46px; height: 46px;
            color: var(--midnight); background: var(--lime); font-family: var(--mono);
            font-size: .6rem; font-weight: 800; letter-spacing: -.05em; transform: rotate(-3deg);
        }
        .surface-brand__mark::after {
            content: ""; position: absolute; left: 50%; top: -6px; width: 1px; height: 58px;
            background: rgba(14,34,60,.32); transform: rotate(28deg);
        }
        .surface-brand__name { font-family: var(--body); font-size: 1rem; font-weight: 800; letter-spacing: -.04em; }
        .surface-brand__name span { color: var(--lime); }
        .surface-brand__meta {
            margin: .75rem 0 0 3.65rem; color: rgba(255,255,255,.45); font-family: var(--mono);
            font-size: .52rem; letter-spacing: .13em; text-transform: uppercase;
        }

        .surface-hero {
            position: relative; display: grid; grid-template-columns: minmax(0, 1.17fr) minmax(330px, .83fr);
            min-height: 690px; overflow: hidden; margin: .5rem 0 0; border: 1px solid var(--ink);
            background: var(--paper-high); box-shadow: 14px 14px 0 var(--ink);
        }
        .surface-hero__copy {
            position: relative; z-index: 3; display: flex; flex-direction: column; justify-content: center;
            min-width: 0; padding: clamp(2rem, 4.5vw, 4.2rem);
        }
        .surface-kicker {
            display: flex; align-items: center; gap: .8rem; margin-bottom: 1.4rem;
            color: var(--cobalt); font-family: var(--mono); font-size: .63rem;
            font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
        }
        .surface-kicker::before { content: ""; width: 11px; height: 11px; background: var(--signal); }
        .surface-hero h1 {
            max-width: 760px; margin: 0; color: var(--ink); font-family: var(--display);
            font-size: clamp(4.2rem, 5.4vw, 6.1rem); font-weight: 400; line-height: .88;
            letter-spacing: -.05em; text-wrap: balance; overflow-wrap: normal; word-break: normal;
            hyphens: none; min-width: 0;
            animation: field-title 720ms var(--ease) both;
        }
        .surface-hero__description {
            max-width: 560px; margin: 1.65rem 0 0; padding-left: 1rem; border-left: 3px solid var(--signal);
            color: var(--ink-soft); font-size: 1.02rem; line-height: 1.65;
        }
        .surface-status {
            display: inline-flex; align-items: center; align-self: flex-start; gap: .6rem;
            margin-top: 1.6rem; padding: .58rem .72rem; border: 1px solid var(--ink);
            color: var(--ink); background: var(--lime); font-family: var(--mono);
            font-size: .57rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
        }
        .surface-status i { width: 7px; height: 7px; border-radius: 50%; background: var(--ink); }
        .surface-hero__visual {
            position: relative; min-width: 0; overflow: hidden; border-left: 1px solid var(--ink);
            background: var(--midnight);
        }
        .surface-hero__image {
            position: absolute; inset: 0; background: url('/app/static/road-scan-hero.webp') 63% 50% / cover no-repeat;
            filter: saturate(.74) contrast(1.12); transform: scale(1.02);
        }
        .surface-hero__visual::before {
            content: "FIELD SCAN / D40"; position: absolute; z-index: 4; top: 1.35rem; left: 1.35rem;
            padding: .48rem .6rem; color: var(--paper-high); background: var(--cobalt);
            font-family: var(--mono); font-size: .53rem; letter-spacing: .13em;
        }
        .surface-hero__visual::after {
            content: ""; position: absolute; z-index: 2; inset: 0;
            background: linear-gradient(180deg, rgba(14,34,60,.06), rgba(14,34,60,.5));
        }
        .surface-lock {
            position: absolute; z-index: 5; left: 20%; top: 43%; width: 58%; aspect-ratio: 1.35;
            border: 2px solid var(--lime); box-shadow: 0 0 0 1px rgba(14,34,60,.35);
        }
        .surface-lock::before, .surface-lock::after { content: ""; position: absolute; width: 22px; height: 22px; }
        .surface-lock::before { left: -4px; top: -4px; border-left: 6px solid var(--lime); border-top: 6px solid var(--lime); }
        .surface-lock::after { right: -4px; bottom: -4px; border-right: 6px solid var(--lime); border-bottom: 6px solid var(--lime); }
        .surface-lock__label {
            position: absolute; left: -2px; bottom: calc(100% + 2px); padding: .34rem .5rem;
            color: var(--midnight); background: var(--lime); font-family: var(--mono);
            font-size: .5rem; font-weight: 800; letter-spacing: .1em; white-space: nowrap;
        }
        .surface-lock__cross { position: absolute; left: 50%; top: 50%; width: 24px; height: 24px; transform: translate(-50%,-50%); }
        .surface-lock__cross::before, .surface-lock__cross::after { content: ""; position: absolute; background: var(--lime); }
        .surface-lock__cross::before { left: 11px; top: 0; width: 2px; height: 24px; }
        .surface-lock__cross::after { left: 0; top: 11px; width: 24px; height: 2px; }
        .surface-sweep {
            position: absolute; z-index: 4; left: 0; right: 0; top: 8%; height: 2px;
            background: linear-gradient(90deg, transparent, var(--lime), transparent);
            box-shadow: 0 0 14px rgba(201,242,75,.6); animation: field-sweep 4.8s ease-in-out infinite;
        }

        .surface-brief {
            display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, .75fr);
            margin: 3.2rem 0 5.5rem; color: white; background: var(--cobalt);
        }
        .surface-brief__statement { position: relative; padding: clamp(2rem, 4.5vw, 4rem); }
        .surface-brief__statement::before {
            content: "“"; position: absolute; right: 2rem; top: .2rem; color: rgba(255,255,255,.12);
            font-family: var(--display); font-size: 9rem; line-height: 1;
        }
        .surface-brief__statement strong {
            position: relative; display: block; max-width: 760px; font-family: var(--display);
            font-size: clamp(2rem, 3.7vw, 3.8rem); font-weight: 400; line-height: 1.02; letter-spacing: -.035em;
        }
        .surface-brief__detail {
            display: flex; align-items: end; padding: clamp(2rem, 3vw, 3rem);
            color: var(--midnight); background: var(--lime);
        }
        .surface-brief__detail p { max-width: 40ch; margin: 0; font-weight: 650; line-height: 1.6; }

        .surface-intro {
            position: relative; overflow: hidden; margin: .5rem 0 3rem; padding: clamp(2.5rem, 6vw, 5.8rem);
            border: 1px solid var(--ink); background: var(--paper-high); box-shadow: 10px 10px 0 var(--cobalt);
        }
        .surface-intro::before {
            content: ""; position: absolute; right: -4rem; top: -7rem; width: 28rem; height: 28rem;
            border: 4.5rem solid rgba(22,70,160,.07); border-radius: 50%;
        }
        .surface-intro::after { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 12px; background: var(--signal); }
        .surface-intro__index { position: relative; color: var(--cobalt); font-family: var(--mono); font-size: .6rem; font-weight: 700; letter-spacing: .15em; text-transform: uppercase; }
        .surface-intro h1 {
            position: relative; max-width: 980px; margin: 1rem 0 .8rem; color: var(--ink); font-family: var(--display);
            font-size: clamp(3.7rem, 6.4vw, 6.5rem); font-weight: 400; line-height: .9;
            letter-spacing: -.05em; text-wrap: balance; overflow-wrap: anywhere; min-width: 0;
            animation: field-title 620ms var(--ease) both;
        }
        .surface-intro p { position: relative; max-width: 670px; margin: 0; color: var(--ink-soft); line-height: 1.65; }
        .surface-intro__status {
            position: absolute; right: 2rem; top: 2rem; padding: .42rem .58rem; border: 1px solid var(--ink);
            color: var(--ink); background: var(--lime); font-family: var(--mono); font-size: .5rem;
            font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
        }
        .surface-intro__status b { color: var(--signal); }

        .surface-section {
            display: grid; grid-template-columns: minmax(0, 1fr) minmax(260px, .55fr); gap: 2rem;
            align-items: end; margin: 4.5rem 0 1.35rem; padding: 0 0 1.15rem; border-bottom: 3px solid var(--ink);
        }
        .surface-section strong { color: var(--ink); font-family: var(--display); font-size: clamp(2.2rem, 3.5vw, 3.5rem); font-weight: 400; line-height: 1; letter-spacing: -.04em; overflow-wrap: anywhere; }
        .surface-section span { max-width: 52ch; color: var(--ink-soft); font-size: .88rem; line-height: 1.55; }

        .st-key-image_action, .st-key-video_action {
            position: relative; min-height: 410px; overflow: hidden; padding: 2rem;
            border: 1px solid var(--ink); border-radius: 2px; cursor: pointer;
            transition: transform 200ms var(--ease), box-shadow 200ms var(--ease);
        }
        .st-key-image_action { color: white; background: var(--cobalt); box-shadow: 9px 9px 0 var(--ink); }
        .st-key-video_action { color: var(--ink); background: var(--signal); box-shadow: 9px 9px 0 var(--ink); }
        .st-key-image_action::after, .st-key-video_action::after {
            content: ""; position: absolute; right: -4rem; top: -5rem; width: 15rem; height: 15rem;
            border: 3.5rem solid currentColor; border-radius: 50%; opacity: .1;
        }
        .st-key-image_action:hover, .st-key-video_action:hover { transform: translate(-3px,-3px); box-shadow: 14px 14px 0 var(--ink); }
        .action-type { display: inline-flex; padding: .36rem .5rem; border: 1px solid currentColor; font-family: var(--mono); font-size: .56rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; }
        .st-key-image_action h3, .st-key-video_action h3 {
            position: relative; z-index: 1; max-width: 430px; margin-top: 5.2rem;
            font-family: var(--display); font-size: clamp(2.6rem, 4vw, 4rem); font-weight: 400;
            line-height: .95; letter-spacing: -.04em; overflow-wrap: anywhere;
        }
        .st-key-image_action p, .st-key-video_action p { position: relative; z-index: 1; max-width: 400px; color: inherit; opacity: .78; }
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

        .surface-media-stamp { display: flex; flex-wrap: wrap; gap: .55rem; margin: .7rem 0 1.1rem; }
        .surface-media-stamp span { padding: .38rem .55rem; border: 1px solid var(--line); color: var(--ink-soft); background: var(--paper-high); font-family: var(--mono); font-size: .52rem; letter-spacing: .07em; text-transform: uppercase; }
        .surface-media-stamp span:first-child { color: white; background: var(--cobalt); border-color: var(--cobalt); }
        .scan-banner { position: relative; overflow: hidden; margin: .75rem 0; padding: 1rem 1rem 1rem 3.4rem; border: 1px solid var(--cobalt); color: var(--ink); background: rgba(22,70,160,.07); font-family: var(--mono); font-size: .62rem; letter-spacing: .05em; }
        .scan-banner::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 2.5rem; background: var(--cobalt); }
        .scan-banner::after { content: ""; position: absolute; left: 2.5rem; top: 0; width: 22%; height: 100%; background: linear-gradient(90deg, rgba(22,70,160,.2), transparent); animation: field-banner 1.9s linear infinite; }
        .result-arrive { height: 4px; margin: 1.5rem 0 0; background: linear-gradient(90deg, var(--signal), var(--lime), transparent 70%); animation: field-result 500ms ease-out both; }

        div[data-testid="stMetric"] { min-height: 132px; padding: 1.05rem 1.15rem; border: 1px solid var(--ink); border-radius: 2px; background: var(--paper-high); box-shadow: 5px 5px 0 var(--paper-low); }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color: var(--cobalt); font-family: var(--mono); font-size: .58rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--ink); font-family: var(--display); letter-spacing: -.035em; }
        div[data-testid="stMetric"]::after { content: ""; display: block; width: 28px; height: 4px; margin-top: .75rem; background: var(--signal); }
        [data-testid="stFileUploaderDropzone"] { min-height: 160px; border: 2px dashed var(--cobalt); border-radius: 2px; background: rgba(255,253,247,.7); }
        [data-testid="stFileUploaderDropzone"]:hover { background: rgba(22,70,160,.07); }
        [data-testid="stForm"] { border: 1px solid var(--line); border-radius: 2px; background: rgba(255,253,247,.76); }
        [data-testid="stImage"] img, [data-testid="stVideo"] video { border: 1px solid var(--ink); box-shadow: 7px 7px 0 var(--paper-low); }
        .stButton button, .stDownloadButton button, [data-testid="stPageLink"] a { border-radius: 2px; font-weight: 750; cursor: pointer; white-space: nowrap; transition: transform 160ms var(--ease), box-shadow 160ms var(--ease); }
        .stButton button:hover, .stDownloadButton button:hover, [data-testid="stPageLink"] a:hover { transform: translate(-2px,-2px); box-shadow: 4px 4px 0 var(--ink); }
        button:focus-visible, a:focus-visible, [role="tab"]:focus-visible, input:focus-visible, [data-testid="stFileUploaderDropzone"]:focus-within { outline: 3px solid var(--cobalt) !important; outline-offset: 3px; }
        .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { color: white; background: var(--signal-dark); border-color: var(--signal-dark); }
        [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 2px solid var(--ink); }
        [data-baseweb="tab"] { border-radius: 2px 2px 0 0; font-family: var(--mono); font-size: .61rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }
        [data-testid="stDataFrame"] { border: 1px solid var(--ink); }

        @keyframes field-title { from { opacity: 0; transform: translateY(38px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes field-sweep { 0%, 8% { top: 7%; opacity: 0; } 18%, 82% { opacity: .9; } 94%, 100% { top: 93%; opacity: 0; } }
        @keyframes field-banner { from { transform: translateX(-120%); } to { transform: translateX(580%); } }
        @keyframes field-result { from { opacity: 0; transform: scaleX(.08); transform-origin: left; } to { opacity: 1; transform: scaleX(1); transform-origin: left; } }

        @media (max-width: 1100px) {
            .surface-hero { grid-template-columns: minmax(0, 1fr); box-shadow: 9px 9px 0 var(--ink); }
            .surface-hero__copy { min-height: 570px; }
            .surface-hero__visual { min-height: 430px; border-left: 0; border-top: 1px solid var(--ink); }
            .surface-hero h1 { font-size: clamp(4.3rem, 12vw, 7rem); }
            .surface-brief { grid-template-columns: minmax(0, 1fr); }
            .surface-section { grid-template-columns: minmax(0, 1fr); gap: .6rem; }
        }
        @media (max-width: 768px) {
            .block-container { padding: .7rem 1rem 4rem; }
            .surface-hero__copy { min-height: 520px; padding: 2rem 1.45rem; }
            .surface-hero h1 { font-size: clamp(3.7rem, 15vw, 5.6rem); }
            .surface-hero__visual { min-height: 360px; }
            .surface-brief { margin-top: 2rem; margin-bottom: 4rem; }
            .surface-intro { padding: 3rem 1.5rem; box-shadow: 7px 7px 0 var(--cobalt); }
            .surface-intro h1 { font-size: clamp(3.2rem, 14vw, 5rem); }
            .surface-intro__status { position: static; display: inline-flex; margin-top: 1.5rem; }
            .st-key-image_action, .st-key-video_action { min-height: 360px; }
        }
        @media (max-width: 480px) {
            .surface-hero { box-shadow: 6px 6px 0 var(--ink); }
            .surface-hero__copy { min-height: 500px; }
            .surface-hero h1 { font-size: clamp(3.35rem, 16.5vw, 4.6rem); }
            .surface-hero__description { font-size: .94rem; }
            .surface-hero__visual { min-height: 320px; }
            .surface-lock { left: 16%; width: 66%; }
            .surface-brief__statement, .surface-brief__detail { padding: 1.6rem; }
            .surface-brief__statement strong { font-size: 2.25rem; }
            .surface-section { margin-top: 3.25rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition: none !important; }
        }
        </style>
        """)


def brand() -> None:
    """Render the application mark in the sidebar."""

    st.html("""
        <div class="surface-brand">
          <div class="surface-brand__row">
            <span class="surface-brand__mark">S/01</span>
            <span class="surface-brand__name">SURFACE<span>/01</span></span>
          </div>
          <div class="surface-brand__meta">Road evidence studio</div>
        </div>
        """)


def hero(kicker: str, title: str, description: str, ready: bool | None = None) -> None:
    """Render the split editorial road-inspection hero."""

    model_label = "Model online" if ready else "Model setup required"
    model_status = ""
    if ready is not None:
        model_status = (
            '<div class="surface-status"><i></i>' f"{html.escape(model_label)}</div>"
        )
    st.html(f"""
        <section class="surface-hero">
          <div class="surface-hero__copy">
            <div class="surface-kicker">{html.escape(kicker)}</div>
            <h1>{html.escape(title)}</h1>
            <p class="surface-hero__description">{html.escape(description)}</p>
            {model_status}
          </div>
          <div class="surface-hero__visual">
            <div class="surface-hero__image"></div>
            <div class="surface-lock" aria-hidden="true">
              <span class="surface-lock__label">POTHOLE CANDIDATE</span>
              <i class="surface-lock__cross"></i>
            </div>
            <div class="surface-sweep" aria-hidden="true"></div>
          </div>
        </section>
        """)


def hero_brief() -> None:
    """Render one concise product statement under the home hero."""

    st.html("""
        <div class="surface-brief">
          <div class="surface-brief__statement">
            <strong>Road media becomes evidence you can inspect—not a verdict you have to trust blindly.</strong>
          </div>
          <div class="surface-brief__detail">
            <p>Image and video input. Three inspection profiles. PyTorch-native inference. Raw uploads stay out of saved history.</p>
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
          <div class="surface-intro__status"><b>●</b>&nbsp; system ready</div>
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
