/* 담당자 화면 잠금 (전국 지도 index.html). 시민 화면(local.html)에는 넣지 않음.
   말벌 신고 앱 검수 화면과 같은 Supabase 계정을 씀 — 같은 주소(karl7208.github.io)라 한쪽에서 로그인하면 다른 쪽도 열림.
   아이디만 넣으면 아이디@hornet.local 계정으로 로그인 (예: admin).
   주의: 화면 잠금임. 지수 자료 파일(data/)은 공개 집계 자료라 그대로 공개. 신고 원자료는 Supabase 권한(RLS)으로 막혀 있음. */
(function () {
  const SUPA_URL = 'https://wutwfnhtpvxvvgvuwker.supabase.co';
  const SUPA_ANON = 'sb_publishable_OduH1zWwcADFq9H3IwQ2sw_9s9fR7Hv';   // 공개용 anon 키 (신고 앱 common.js와 같음, 비밀 아님)
  const STAFF_DOMAIN = 'hornet.local';
  if (window.HR_GATE_OFF) return;   // 논문 그림 캡처(오프라인)용
  const q = new URLSearchParams(location.search).get('lang');
  let lang = q; if (lang !== 'en' && lang !== 'ko') { try { lang = localStorage.getItem('hr.lang'); } catch { lang = null; } }
  const EN = lang === 'en';
  const t = (ko, en) => EN ? en : ko;
  const css = document.createElement('style');
  css.textContent = `#hrGate{position:fixed;inset:0;z-index:10000;background:var(--bg,#EDF0EB);display:flex;align-items:center;justify-content:center;padding:16px}
#hrGate form{width:100%;max-width:340px;background:var(--surface,#fff);color:var(--ink,#1B2420);border:1px solid var(--line,#D3DAD4);border-radius:12px;padding:20px;font-family:system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}
#hrGate h2{margin:0 0 4px;font-size:1.05rem} #hrGate p{margin:0 0 12px;font-size:.84rem;color:var(--muted,#5C6962);line-height:1.5}
#hrGate input{width:100%;padding:10px;border:1px solid var(--line,#D3DAD4);border-radius:7px;font:inherit;margin-bottom:8px;background:var(--surface,#fff);color:inherit}
#hrGate button{width:100%;padding:10px;border:0;border-radius:7px;background:var(--pine,#22463A);color:#fff;font:inherit;font-weight:600;cursor:pointer}
#hrGate .msg{color:#9E3F27;font-size:.84rem;min-height:1.2em;margin-top:8px} #hrGate a{color:inherit}`;
  document.head.appendChild(css);
  const g = document.createElement('div'); g.id = 'hrGate';
  g.innerHTML = `<form autocomplete="on"><h2>${t('담당자 로그인', 'Staff login')}</h2>
<p>${t('전국 지도는 담당자용입니다. 시민께서는 <a href="local.html">우리 지역 보기</a>를 이용해 주세요.', 'The nationwide map is for staff. Citizens, please use <a href="local.html?lang=en">My area</a>.')}</p>
<input name="uid" type="text" placeholder="${t('아이디', 'ID')}" autocomplete="username" autocapitalize="off" spellcheck="false" required>
<input name="pw" type="password" placeholder="${t('비밀번호', 'Password')}" autocomplete="current-password" required>
<button type="submit">${t('로그인', 'Log in')}</button><div class="msg" aria-live="polite">${t('확인 중…', 'Checking…')}</div></form>`;
  document.body.appendChild(g);
  const f = g.querySelector('form'), msg = g.querySelector('.msg');
  let client = null;
  const open = () => {
    g.remove();
    const hb = document.getElementById('langBtn');
    if (hb && !document.getElementById('hrLogout')) {
      const b = document.createElement('button'); b.id = 'hrLogout'; b.type = 'button'; b.className = hb.className; b.style.cssText = hb.style.cssText; b.style.marginLeft = '6px';
      b.textContent = t('로그아웃', 'Log out');
      b.onclick = async () => { try { await client.auth.signOut(); } catch {} location.reload(); };
      hb.after(b);
    }
  };
  const load = src => new Promise((ok, no) => { const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = () => no(new Error('load')); document.head.appendChild(s); });
  (async () => {
    try {
      if (!window.supabase) await load('https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/dist/umd/supabase.js');
      client = window.supabase.createClient(SUPA_URL, SUPA_ANON);
      const r = await client.auth.getSession();
      if (r.data && r.data.session) return open();
      msg.textContent = '';
      f.elements.uid.focus();
    } catch { msg.textContent = t('로그인 서버에 연결하지 못했습니다. 잠시 뒤 새로고침해 주세요.', 'Could not reach the login server. Please reload later.'); }
  })();
  f.onsubmit = async e => {
    e.preventDefault(); if (!client) return;
    msg.textContent = '';
    const id = f.elements.uid.value.trim(); const email = id.includes('@') ? id : id.toLowerCase() + '@' + STAFF_DOMAIN;
    const r = await client.auth.signInWithPassword({ email, password: f.elements.pw.value });
    if (r.error) { msg.textContent = r.error.message === 'Invalid login credentials' ? t('아이디 또는 비밀번호가 맞지 않습니다', 'Wrong ID or password') : r.error.message; return; }
    open();
  };
})();
