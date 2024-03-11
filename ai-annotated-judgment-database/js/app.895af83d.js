(function(){"use strict";var e={1441:function(e,t,a){var n=a(9242),l=a(3396),o=a(7139);const s={class:"main-container"};function r(e,t,a,n,r,i){const c=(0,l.up)("router-view"),u=(0,l.up)("my-footer");return(0,l.wg)(),(0,l.iD)(l.HY,null,[(0,l._)("div",s,[(0,l._)("div",{class:"jump-bar",style:(0,o.j5)(i.backgroundStyle)},null,4),(0,l.Wm)(c)]),(0,l.Wm)(u)],64)}var i=a.p+"img/logo_nthu.e807351d.png",c=a.p+"img/icon.c0426136.jpg";const u=e=>((0,l.dD)("data-v-44fdd587"),e=e(),(0,l.Cn)(),e),d={class:"footer"},p=u((()=>(0,l._)("div",{class:"container"},[(0,l._)("span",null,"聯絡我們：custodyprediction@gmail.com"),(0,l._)("span",null,"©copyright Artificial Intelligence for Fundamental Research (AIFR) Group"),(0,l._)("div",null,[(0,l._)("img",{src:i,width:"200",height:"40",class:"d-inline-block align-top",style:{margin:"0 5px"},alt:"",loading:"lazy"}),(0,l._)("img",{src:c,width:"80",height:"45",class:"d-inline-block align-top",style:{margin:"0 5px"},alt:"",loading:"lazy"})])],-1))),m=[p];function h(e,t,a,n,o,s){return(0,l.wg)(),(0,l.iD)("footer",d,m)}var g={name:"my-footer",data(){return{backend:[],isLoading:!1,errorPrompt:!1,errorCode:""}},props:{},methods:{}},f=a(89);const y=(0,f.Z)(g,[["render",h],["__scopeId","data-v-44fdd587"]]);var _=y,w={components:{myFooter:_},data(){return{page_name:"搜尋頁面"}},computed:{backgroundStyle(){const e=a(4148);return{backgroundImage:`url(${e})`}}}};const v=(0,f.Z)(w,[["render",r]]);var b=v,D=a(2483);const C={class:"container mt-4"},k={class:"form-group"},x=(0,l._)("h6",null,"法院別",-1),S=(0,l._)("label",{for:"selectAll"},"所有法院",-1),z=(0,l._)("label",{for:"selectSupremeCourts"},"所有高等法院",-1),j=(0,l._)("label",{for:"selectDistrictCourts"},"所有地方法院",-1),T=(0,l._)("div",{class:"smartphone-message"},"請點擊下方輸入框以選取法院",-1),R={class:"form-group"},O=(0,l._)("div",{class:"mt-3"},[(0,l._)("strong",null,"裁判日期")],-1),H=(0,l._)("span",{class:"p-2"},"起",-1),V=(0,l._)("span",null,"民國",-1),I=(0,l._)("option",{value:111},"111",-1),q=[I],M=(0,l._)("span",null,"年",-1),F=["value"],U=(0,l._)("span",null,"月",-1),A=(0,l._)("span",{class:"p-2"},"迄",-1),P=(0,l._)("span",null,"民國",-1),Y=(0,l._)("option",{value:111},"111",-1),E=[Y],L=(0,l._)("span",null,"年",-1),W=["value"],$=(0,l._)("span",null,"月",-1),K={class:"p-0 border rounded-3",style:{overflow:"scroll"}},Z={class:"table table-bordered custom-adjust-table"},Q=(0,l._)("thead",{class:"text-center"},[(0,l._)("tr",null,[(0,l._)("th",{style:{width:"350px","background-color":"#f5f5f5"}},"搜尋類型"),(0,l._)("th",{style:{"min-width":"350px"},class:"custom-light-purple"},"請輸入搜尋條件")])],-1),G={style:{"line-height":"50px","padding-left":"30px !important","background-color":"#f5f5f5"}},B={class:"custom-light-purple"},J=["onUpdate:modelValue","placeholder"],N={style:{"background-color":"#f5f5f5"}},X=["id","value"],ee=["for"],te=(0,l._)("div",{class:"form-instruction"},"💡此處只能選擇見解，心證，或涵攝其中一項",-1),ae={class:"custom-light-purple"},ne=["placeholder"],le={class:"d-flex flex-row-reverse my-5"},oe={key:0,class:"alert alert-danger mt-2",role:"alert"};function se(e,t,a,s,r,i){const c=(0,l.up)("el-col"),u=(0,l.up)("el-row");return(0,l.wg)(),(0,l.iD)("div",C,[(0,l.Wm)(u,null,{default:(0,l.w5)((()=>[(0,l.Wm)(c,{lg:{span:6},md:24},{default:(0,l.w5)((()=>[(0,l._)("form",{onSubmit:t[11]||(t[11]=(0,n.iM)(((...e)=>i.onSubmit&&i.onSubmit(...e)),["prevent"]))},[(0,l._)("div",k,[x,(0,l._)("div",null,[(0,l.wy)((0,l._)("input",{type:"checkbox",id:"selectAll","onUpdate:modelValue":t[0]||(t[0]=e=>r.selectAllCourts=e),onChange:t[1]||(t[1]=(...e)=>i.selectAllChanged&&i.selectAllChanged(...e))},null,544),[[n.e8,r.selectAllCourts]]),S]),(0,l._)("div",null,[(0,l.wy)((0,l._)("input",{type:"checkbox",id:"selectSupremeCourts","onUpdate:modelValue":t[2]||(t[2]=e=>r.selectSupremeCourts=e),onChange:t[3]||(t[3]=(...e)=>i.selectSupremeCourtsChanged&&i.selectSupremeCourtsChanged(...e))},null,544),[[n.e8,r.selectSupremeCourts]]),z]),(0,l._)("div",null,[(0,l.wy)((0,l._)("input",{type:"checkbox",id:"selectDistrictCourts","onUpdate:modelValue":t[4]||(t[4]=e=>r.selectDistrictCourts=e),onChange:t[5]||(t[5]=(...e)=>i.selectDistrictCourtsChanged&&i.selectDistrictCourtsChanged(...e))},null,544),[[n.e8,r.selectDistrictCourts]]),j]),T,(0,l._)("div",null,[(0,l.wy)((0,l._)("select",{class:"form-select",multiple:"",style:{height:"50vh",overflow:"scroll"},"onUpdate:modelValue":t[6]||(t[6]=e=>r.selectedCourts=e)},[((0,l.wg)(!0),(0,l.iD)(l.HY,null,(0,l.Ko)(r.courtTypeOptions,((e,t)=>((0,l.wg)(),(0,l.iD)("option",{key:t},(0,o.zw)(e.name),1)))),128))],512),[[n.bM,r.selectedCourts]])])]),(0,l._)("div",R,[O,(0,l._)("div",null,[H,V,(0,l.wy)((0,l._)("select",{class:"form-select form-select-sm",style:{width:"fit-content",display:"inline"},"onUpdate:modelValue":t[7]||(t[7]=e=>r.selectedDateRange.from.year=e)},q,512),[[n.bM,r.selectedDateRange.from.year]]),M,(0,l.wy)((0,l._)("select",{class:"form-select form-select-sm",style:{width:"fit-content",display:"inline"},"onUpdate:modelValue":t[8]||(t[8]=e=>r.selectedDateRange.from.month=e)},[((0,l.wg)(),(0,l.iD)(l.HY,null,(0,l.Ko)(12,(e=>(0,l._)("option",{value:e,key:e},(0,o.zw)(e),9,F))),64))],512),[[n.bM,r.selectedDateRange.from.month]]),U]),(0,l._)("div",null,[A,P,(0,l.wy)((0,l._)("select",{class:"form-select form-select-sm",style:{width:"fit-content",display:"inline"},"onUpdate:modelValue":t[9]||(t[9]=e=>r.selectedDateRange.to.year=e)},E,512),[[n.bM,r.selectedDateRange.to.year]]),L,(0,l.wy)((0,l._)("select",{class:"form-select form-select-sm",style:{width:"fit-content",display:"inline"},"onUpdate:modelValue":t[10]||(t[10]=e=>r.selectedDateRange.to.month=e)},[((0,l.wg)(),(0,l.iD)(l.HY,null,(0,l.Ko)(12,(e=>(0,l._)("option",{value:e,key:e},(0,o.zw)(e),9,W))),64))],512),[[n.bM,r.selectedDateRange.to.month]]),$])])],32)])),_:1}),(0,l.Wm)(c,{lg:{span:17,offset:1},md:24,style:{"margin-top":"20px"}},{default:(0,l.w5)((()=>[(0,l._)("div",K,[(0,l._)("table",Z,[Q,(0,l._)("tbody",null,[((0,l.wg)(!0),(0,l.iD)(l.HY,null,(0,l.Ko)(r.formData.searchFields,((e,t)=>((0,l.wg)(),(0,l.iD)("tr",{key:t},[(0,l._)("td",G,(0,o.zw)(e.type),1),(0,l._)("td",B,[(0,l.wy)((0,l._)("input",{type:"text",class:"form-control custom-light-purple",style:{height:"50px","background-color":"#fff !important"},"onUpdate:modelValue":t=>e.query=t,placeholder:e.example},null,8,J),[[n.nr,e.query]])])])))),128)),(0,l._)("tr",null,[(0,l._)("td",N,[((0,l.wg)(!0),(0,l.iD)(l.HY,null,(0,l.Ko)(r.poolOptions,(e=>((0,l.wg)(),(0,l.iD)("div",{class:"form-check mx-auto",style:{width:"fit-content"},key:e.name},[(0,l.wy)((0,l._)("input",{class:"form-check-input",type:"radio",name:"flexRadio",id:e.name,"onUpdate:modelValue":t[12]||(t[12]=e=>r.selectedSearchType=e),value:e.name},null,8,X),[[n.G2,r.selectedSearchType]]),(0,l._)("label",{class:"form-check-label",style:{"text-align":"left","margin-left":"10px"},for:e.name},(0,o.zw)(e.type),9,ee)])))),128)),te]),(0,l._)("td",ae,[(0,l.wy)((0,l._)("textarea",{class:"form-control custom-light-purple",style:{height:"130px","background-color":"#fff !important"},"onUpdate:modelValue":t[13]||(t[13]=e=>r.poolOptions[r.selectedSearchType].query=e),placeholder:r.poolOptions[r.selectedSearchType].example},null,8,ne),[[n.nr,r.poolOptions[r.selectedSearchType].query]])])])])])]),(0,l._)("div",le,[(0,l._)("button",{class:"btn btn-secondary d-inline-flex custom-purlpe",onClick:t[14]||(t[14]=(...e)=>i.advanceSearch&&i.advanceSearch(...e))},"進階搜尋條件送出")])])),_:1})])),_:1}),r.showErrorAlert?((0,l.wg)(),(0,l.iD)("div",oe," 無法同時選擇一項以上的涵攝，見解，或心證，請修改後送出。 ")):(0,l.kq)("",!0)])}a(560);var re=a(7178),ie={name:"SearchForm",components:{},data(){return{formData:{court_type:"",refereeDate:"",searchFields:[{type:"案件別",name:"case_kind",example:"例如詐欺",query:""},{type:"當事人等基本資料",name:"basic_info",example:"",query:""},{type:"主文中的關鍵字",name:"syllabus",example:"",query:""},{type:"判決全文的關鍵字",name:"jud_full",example:"",query:""}]},selectedSearchType:"opinion",poolOptions:{opinion:{type:"法院見解的關鍵字",name:"opinion",query:"",example:"請輸入法院見解的關鍵字"},fee:{type:"法官心證的關鍵字(限地院)",name:"fee",query:"",example:"請輸入法官心證的關鍵字(限地院)"},sub:{type:"法官涵攝的關鍵字(限地院)",name:"sub",query:"",example:"請輸入法官涵攝的關鍵字(限地院)"}},courtTypeOptions:[{name:"最高法院",value:"zgf"},{name:"臺灣高等法院",value:"twgdfy"},{name:"智慧財產及商業法院",value:"zhccjsyfy"},{name:"臺灣高等法院臺中分院",value:"twgdfytcfy"},{name:"臺灣高等法院臺南分院",value:"twgdfytnfy"},{name:"臺灣高等法院高雄分院",value:"twgdfykxfy"},{name:"臺灣高等法院花蓮分院",value:"twgdfyhlfy"},{name:"福建高等法院金門分院",value:"fjgdfyjmfy"},{name:"臺灣臺北地方法院",value:"twtbdfy"},{name:"臺灣新北地方法院",value:"twxbdfy"},{name:"臺灣士林地方法院",value:"twslgdfy"},{name:"臺灣桃園地方法院",value:"twtydfy"},{name:"臺灣新竹地方法院",value:"twxzdfy"},{name:"臺灣苗栗地方法院",value:"twmldfy"},{name:"臺灣臺中地方法院",value:"twtcdfy"},{name:"臺灣南投地方法院",value:"twntdfy"},{name:"臺灣彰化地方法院",value:"twzhdfy"},{name:"臺灣雲林地方法院",value:"twyldfy"},{name:"臺灣嘉義地方法院",value:"twjydfy"},{name:"臺灣臺南地方法院",value:"twtndfy"},{name:"臺灣高雄地方法院",value:"twkxdfy"},{name:"臺灣橋頭地方法院",value:"twqtdfy"},{name:"臺灣屏東地方法院",value:"twptdfy"},{name:"臺灣臺東地方法院",value:"twtdgdfy"},{name:"臺灣花蓮地方法院",value:"twhldfy"},{name:"臺灣宜蘭地方法院",value:"twyldfy"},{name:"臺灣基隆地方法院",value:"twjldfy"},{name:"臺灣澎湖地方法院",value:"twphdfy"},{name:"福建金門地方法院",value:"fjjmdfy"},{name:"福建連江地方法院",value:"fjljdfy"}],selectedCourts:[],selectAllCourts:!1,selectSupremeCourts:!1,selectDistrictCourts:!1,showModal:!1,isSelectedAllCourts:!0,selectedDateRange:{from:{year:"111",month:"1"},to:{year:"111",month:"12"}},showErrorAlert:!1}},mounted(){this.initializeForm()},watch:{"selectedDateRange.from":{handler(e){const t=parseInt(e.year),a=parseInt(e.month),n=parseInt(this.selectedDateRange.to.year),l=parseInt(this.selectedDateRange.to.month);(t>n||t===n&&a>l)&&(this.selectedDateRange.to.year=e.year,this.selectedDateRange.to.month=e.month),(0,re.z8)({message:"起始日不可晚於結束日",type:"warning"})},deep:!0},"selectedDateRange.to":{handler(e){const t=parseInt(e.year),a=parseInt(e.month),n=parseInt(this.selectedDateRange.from.year),l=parseInt(this.selectedDateRange.from.month);(t<n||t===n&&a<l)&&(this.selectedDateRange.from.year=e.year,this.selectedDateRange.from.month=e.month)},deep:!0},selectedCourts(e){this.selectAllCourts=e.length===this.courtTypeOptions.length}},methods:{getSelectableYears(){return(new Date).getFullYear()-1911},formatPart(e){return String(e).padStart(2,"0")},getLastDayOfMonth(e,t){let a=new Date(e,t+1,0);return a.getDate()},dateFormat(){const e=1911+parseInt(this.selectedDateRange.from.year),t=String(this.selectedDateRange.from.month),a=1911+parseInt(this.selectedDateRange.to.year),n=String(this.selectedDateRange.to.month);let l=this.getLastDayOfMonth(a,n-1);return`${e}${t.padStart(2,"0")}01-${a}${n.padStart(2,"0")}${l}`},selectAllChanged(){this.selectAllCourts?this.selectedCourts=this.courtTypeOptions.map((e=>e.name)):this.selectedCourts=[]},selectSupremeCourtsChanged(){const e=this.courtTypeOptions.filter((e=>e.name.includes("高等法院"))).map((e=>e.name));this.updateSelection(this.selectSupremeCourts,e)},selectDistrictCourtsChanged(){const e=this.courtTypeOptions.filter((e=>e.name.includes("地方法院"))).map((e=>e.name));this.updateSelection(this.selectDistrictCourts,e)},updateSelection(e,t){this.selectedCourts=e?[...new Set([...this.selectedCourts,...t])]:this.selectedCourts.filter((e=>!t.includes(e)))},initializeForm(){this.formData.court_type=this.courtTypeOptions.map((e=>e.name)).join(" ")},onSubmit(){console.log("Submitted",this.formData)},showBootstrapWarning(){this.showErrorAlert=!0,setTimeout((()=>{this.showErrorAlert=!1}),5e3)},advanceSearch(){let e={};this.formData.court_type=this.selectedCourts.join(" "),this.formData.refereeDate=this.dateFormat(),e.court_type=this.formData.court_type,e.jud_date=this.formData.refereeDate,this.formData.searchFields.forEach((t=>{e[t.name]=t.query})),""!=this.selectedSearchType&&(e[this.selectedSearchType]=this.poolOptions[this.selectedSearchType].query),this.$router.push({path:"/search-result",query:e})}}};const ce=(0,f.Z)(ie,[["render",se]]);var ue=ce;const de={class:"container"},pe={class:"row"},me={class:"col-md-9"},he=(0,l._)("div",{class:"fw-bolder my-1 py-1"},"搜尋條件",-1),ge={class:"d-flex flex-wrap"},fe={class:"col-md-3"},ye=(0,l._)("div",{class:"fw-bolder my-1 py-1"},"查詢結果",-1),_e={class:"text-decoration-underline"},we={class:"pagination-container"},ve={class:"row mt-3 mb-3"},be={class:"col-md-12"},De={class:"rounded-3 border"},Ce={class:"table table-bordered custom-adjust-table"},ke={class:"text-center"},xe=(0,l._)("th",{class:"custom-blue"},"Index",-1),Se={key:0,class:"custom-blue"},ze={style:{"text-align":"center"}},je=["href"],Te={key:0,style:{color:"rgb(138, 138, 138)"}},Re=["onClick"],Oe=["innerHTML"],He=(0,l._)("span",{class:"tooltiptext"},"點擊閱讀全文",-1),Ve=["innerHTML"],Ie={key:0},qe=(0,l._)("td",{class:"no-found-cell",colspan:"8"},"查無資料 ",-1),Me=[qe],Fe=["innerHTML"];function Ue(e,t,a,n,s,r){const i=(0,l.up)("router-link"),c=(0,l.up)("el-pagination"),u=(0,l.up)("el-dialog"),d=(0,l.Q2)("loading");return(0,l.wy)(((0,l.wg)(),(0,l.iD)("div",de,[(0,l._)("div",pe,[(0,l._)("div",me,[(0,l.Wm)(i,{to:"/"},{default:(0,l.w5)((()=>[(0,l.Uk)("回搜尋頁")])),_:1}),he,(0,l._)("div",null,[(0,l._)("div",ge,[((0,l.wg)(!0),(0,l.iD)(l.HY,null,(0,l.Ko)(s.conditionInfo,(e=>((0,l.wg)(),(0,l.iD)("div",{class:"px-2 pb-1",key:e},(0,o.zw)(r.getConditionInfo(e)),1)))),128))])])]),(0,l._)("div",fe,[ye,((0,l.wg)(!0),(0,l.iD)(l.HY,null,(0,l.Ko)(s.resultCount,(e=>((0,l.wg)(),(0,l.iD)("div",{key:e.name},[0!=e.count?((0,l.wg)(),(0,l.iD)(l.HY,{key:0},[(0,l.Uk)(" 共計 "),(0,l._)("strong",_e,(0,o.zw)(e.count),1),(0,l.Uk)(" "+(0,o.zw)(e.unit)+(0,o.zw)(e.name),1)],64)):(0,l.kq)("",!0)])))),128))])]),(0,l._)("div",we,[(0,l.Wm)(c,{"current-page":s.pageDetial.page,"page-size":s.pageDetial.size,"page-sizes":[10,50,100,200],total:s.pageDetial.total,background:"",layout:"total, sizes, prev, pager, next",onSizeChange:r.handlePageSize,onCurrentChange:r.handlePageChange},null,8,["current-page","page-size","total","onSizeChange","onCurrentChange"]),(0,l._)("div",ve,[(0,l._)("div",be,[(0,l._)("div",De,[(0,l._)("table",Ce,[(0,l._)("thead",null,[(0,l._)("tr",ke,[xe,((0,l.wg)(!0),(0,l.iD)(l.HY,null,(0,l.Ko)(s.searchFields,(e=>((0,l.wg)(),(0,l.iD)(l.HY,{key:e.name},[r.checkQueryEnable(e.name)?((0,l.wg)(),(0,l.iD)("th",Se,(0,o.zw)(e.type),1)):(0,l.kq)("",!0)],64)))),128))])]),(0,l._)("tbody",null,[((0,l.wg)(!0),(0,l.iD)(l.HY,null,(0,l.Ko)(s.searchResults,((e,t)=>((0,l.wg)(),(0,l.iD)("tr",{key:t},[(0,l._)("td",ze,(0,o.zw)(t+(s.pageDetial.page-1)*s.pageDetial.size+1),1),((0,l.wg)(!0),(0,l.iD)(l.HY,null,(0,l.Ko)(s.searchFields,(t=>((0,l.wg)(),(0,l.iD)(l.HY,{key:t.name},[r.checkQueryEnable(t.name)?((0,l.wg)(),(0,l.iD)("td",{key:0,style:(0,o.j5)(r.getColumnWidth(t.name))},["case_num"==t.name?((0,l.wg)(),(0,l.iD)("a",{key:0,href:e["jud_url"],target:"_blank"},(0,o.zw)(e["case_num"]),9,je)):((0,l.wg)(),(0,l.iD)(l.HY,{key:1},[null==e[t.name]?((0,l.wg)(),(0,l.iD)("p",Te,"無")):e[t.name].length>s.maxTextLength||(e[t.name].match(/\n/g)||[]).length>5?((0,l.wg)(),(0,l.iD)("p",{key:1,class:"mytooltip custom-overflow-column",onClick:a=>r.openDialog(r.addHighlighter(t.name,e[t.name]))},[(0,l._)("span",{innerHTML:r.addHighlighter(t.name,e[t.name].substr(0,250))},null,8,Oe),(0,l.Uk)("...more "),He],8,Re)):((0,l.wg)(),(0,l.iD)("p",{key:2,innerHTML:r.addHighlighter(t.name,e[t.name])},null,8,Ve))],64))],4)):(0,l.kq)("",!0)],64)))),128))])))),128)),0==s.searchResults.length?((0,l.wg)(),(0,l.iD)("tr",Ie,Me)):(0,l.kq)("",!0)])])])])]),(0,l.Wm)(u,{modelValue:s.dialogVisible,"onUpdate:modelValue":t[0]||(t[0]=e=>s.dialogVisible=e),width:"60%",style:{"max-height":"80vh",overflow:"scroll"},"before-close":r.handleClose},{default:(0,l.w5)((()=>[(0,l._)("span",{innerHTML:s.dialogText},null,8,Fe)])),_:1},8,["modelValue","before-close"]),(0,l.Wm)(c,{"current-page":s.pageDetial.page,"page-size":s.pageDetial.size,"page-sizes":[10,50,100,200],total:s.pageDetial.total,background:"",layout:"total, sizes, prev, pager, next",onSizeChange:r.handlePageChange,onCurrentChange:r.handlePageChange},null,8,["current-page","page-size","total","onSizeChange","onCurrentChange"])])])),[[d,s.loading]])}a(8858),a(1318),a(3228);var Ae=a(1076),Pe={data(){return{loading:!0,mode:"default",params:{},searchFields:{court_type:{type:"法院",name:"court_type"},case_num:{type:"案號",name:"case_num"},jud_date:{type:"日期",name:"jud_date"},case_type:{type:"案件別",name:"case_type"},basic_info:{type:"當事人等基本資料",name:"basic_info"},opinion:{type:"見解",name:"opinion"},fee:{type:"心證",name:"fee"},sub:{type:"涵攝",name:"sub"},jud_full:{type:"全文關鍵字",name:"jud_full"}},searchResults:[],resultCount:[],pageDetial:{page:1,size:10,next_page_url:"null",previous_page_url:"null",total_pages:8,total:15},conditionInfo:[],maxTextLength:250,dialogVisible:!1,dialogText:""}},created(){this.initParams(),this.fetchData()},methods:{getColumnWidth(e){return"basic_info"==e?"min-width: 260px;":"jud_date"==e?"min-width: 90px;":"case_type"==e?"min-width: 125px;":"case_num"==e?"min-width: 90px;":""},openDialog(e){this.dialogVisible=!0,this.dialogText=e},handleClose(){this.dialogVisible=!1},checkQueryEnable(e){return"court_type"!=e&&("opinion"!=e&&"fee"!==e&&"sub"!==e&&"jud_full"!=e||!!this.params[e])},handlePageSize(e){console.log(e),this.pageDetial.size=e,this.fetchData()},handlePageChange(e){this.pageDetial.page=e,this.fetchData()},getConditionInfo(e){if(this.searchFields[e[0]])return`${this.searchFields[e[0]].type||""}:  ${e[1]}`},addHighlighter(e,t){let a=null;this.conditionInfo.forEach((t=>{t[0]!=e||(a=t[1])}));let n=t;return a&&a.split(" ").forEach((e=>{n=n.replace(e,`<span class="highlighter">${e}</span>`)})),n=n.replace(/\n+/g,"<br>"),n=n.replace(/<br>$/,""),n},removeEmptyStringValues(e){return Object.keys(e).forEach((t=>{""===e[t]&&delete e[t]})),e},initParams(){const e=new URLSearchParams(window.location.search);this.params={search_method:"keyword",page:1,size:this.pageDetial.size,court_type:e.get("court_type")||"",jud_date:e.get("jud_date")||"",basic_info:e.get("basic_info")||"",syllabus:e.get("syllabus")||"",opinion:e.get("opinion")||"",fee:e.get("fee")||"",sub:e.get("sub")||"",jud_full:e.get("jud_full")||""},this.params=this.removeEmptyStringValues(this.params)},async fetchData(){this.loading=!0,this.params.page=this.pageDetial.page,this.params.size=this.pageDetial.size;try{const e=await Ae.Z.get("https://namely-fast-ocelot.ngrok-free.app/api/search",{headers:{"ngrok-skip-browser-warning":"69420"},params:this.params}),t=e.data;this.searchResults=t.data,this.conditionInfo=t.condition_info.available,this.pageDetial=t.meta,this.resultCount=t.summary,this.loading=!1}catch(e){console.error("There was an error!",e),this.loading=!1}}}};const Ye=(0,f.Z)(Pe,[["render",Ue]]);var Ee=Ye;const Le=[{path:"/",name:"home",component:ue},{path:"/search-result",name:"search-result",component:Ee}],We=(0,D.p7)({history:(0,D.PO)("/ai-annotated-judgment-database/"),routes:Le});var $e=We,Ke=a(65),Ze=(0,Ke.MT)({state:{},getters:{},mutations:{},actions:{},modules:{}}),Qe=(a(1800),a(3960)),Ge=a(4553);a(4415);(0,n.ri)(b).use(Ze).use($e).use(Ge.Z).component("VueDatePicker",Qe.Z).mount("#app")},4148:function(e,t,a){e.exports=a.p+"img/裁判觀點檢索平台2黑白.97c81404.png"}},t={};function a(n){var l=t[n];if(void 0!==l)return l.exports;var o=t[n]={exports:{}};return e[n].call(o.exports,o,o.exports,a),o.exports}a.m=e,function(){var e=[];a.O=function(t,n,l,o){if(!n){var s=1/0;for(u=0;u<e.length;u++){n=e[u][0],l=e[u][1],o=e[u][2];for(var r=!0,i=0;i<n.length;i++)(!1&o||s>=o)&&Object.keys(a.O).every((function(e){return a.O[e](n[i])}))?n.splice(i--,1):(r=!1,o<s&&(s=o));if(r){e.splice(u--,1);var c=l();void 0!==c&&(t=c)}}return t}o=o||0;for(var u=e.length;u>0&&e[u-1][2]>o;u--)e[u]=e[u-1];e[u]=[n,l,o]}}(),function(){a.n=function(e){var t=e&&e.__esModule?function(){return e["default"]}:function(){return e};return a.d(t,{a:t}),t}}(),function(){a.d=function(e,t){for(var n in t)a.o(t,n)&&!a.o(e,n)&&Object.defineProperty(e,n,{enumerable:!0,get:t[n]})}}(),function(){a.g=function(){if("object"===typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"===typeof window)return window}}()}(),function(){a.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)}}(),function(){a.r=function(e){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})}}(),function(){a.p="/ai-annotated-judgment-database/"}(),function(){var e={143:0};a.O.j=function(t){return 0===e[t]};var t=function(t,n){var l,o,s=n[0],r=n[1],i=n[2],c=0;if(s.some((function(t){return 0!==e[t]}))){for(l in r)a.o(r,l)&&(a.m[l]=r[l]);if(i)var u=i(a)}for(t&&t(n);c<s.length;c++)o=s[c],a.o(e,o)&&e[o]&&e[o][0](),e[o]=0;return a.O(u)},n=self["webpackChunkai_annotated_judgment_database"]=self["webpackChunkai_annotated_judgment_database"]||[];n.forEach(t.bind(null,0)),n.push=t.bind(null,n.push.bind(n))}();var n=a.O(void 0,[998],(function(){return a(1441)}));n=a.O(n)})();
//# sourceMappingURL=app.895af83d.js.map