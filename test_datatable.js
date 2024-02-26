async function show_query_result(clicked_button_name){
  // domain = 'https://1fb4-112-104-64-172.ngrok-free.app/api/search/all?'
  // domain = 'http://127.0.0.1:8000/api/search/all?'
  domain = 'http://140.114.80.195:6127/api/search/all?'
  query_type = document.getElementsByName('query_type')
  for (var radio of query_type){
    if (radio.checked){
      query_type = radio.value;
    }
  }
  query_text = document.getElementById('query_text').value
  let query_parameters = {
  

    court_type: document.getElementById('court_type').value,
    jud_date: document.getElementById('jud_date').value,
    syllabus: document.getElementById('syllabus').value,
    basic_info: document.getElementById('basic_info').value,
    jud_full: document.getElementById('jud_full').value
  };

  show_query_content = document.getElementById('query_content');
  show_query_content.innerHTML = `${query_text}`;

  request_url = domain + `search_method=${clicked_button_name}`

  for (const key in query_parameters) {
    value = query_parameters[key]
    if (value === '') {
      continue
    }
    else{
      request_url += `&${key}=${value}`
    }
  }
  if (query_text!=''){
    request_url += `&${query_type}=${query_text}`
  }
  document.getElementById('show_input').innerHTML = `<a href="${request_url}">${request_url}</a>`
  let query_types = {
    'sub': '涵攝',
    'opinion': '見解',
    'fee': '心證',
  }
  let api_json;
  const response = await fetch(request_url, {
    method: 'get',
    headers: new Headers({
        "ngrok-skip-browser-warning": "69420"
    })
  });
  const data = await response.json();
  let table_body = ""
  let table_head = `
    <tr>
      <th style="width:3em">院別</th>
      <th style="width:5em">裁判日期</th>
      <th style="width:14em">行政資訊</br>（案號, 日期, 法官, 檢察官）</th>
      <th data-orderable="false">${query_types[query_type]}</th>
    </tr>
  `;
  data.data.forEach((result) => {
    table_body += `
    <tr>
      <td>${result['court_type']}</td>
      <td>${result['jud_date']}</td>
      <td><a href="${result['jud_url']}" data-toggle="tooltip" data-placement="bottom" title="前往判決原文" target="_blank">${result['JID']}</a></td>
      <td><span class="copy-text" data-toggle="tooltip" data-placement="bottom" title="點擊以複製文字">${result[query_type]}</span></td>
    </tr>`;
  });
  document.getElementById('show_input').innerHTML = `<a href="${request_url}">${request_url}</a>` + '</br>' + 
  '符合的條件:</br>' + JSON.stringify(data['condition_info']['available'], undefined, 2) + '</br>' +
  '不符合的條件:</br>' + JSON.stringify(data['condition_info']['unavailable'], undefined, 2);

  document.getElementById('thead').innerHTML = table_head
  document.getElementById('tbody').innerHTML = table_body
  $("#targetTable").dataTable().fnDestroy();
  $('#targetTable').DataTable({
    'data': data.data,
    'columns': [
      {'data': 'court_type'},
      {'data': 'jud_date'},
      {'data': 'JID'},
      {'data': query_type},
    ], 
    "language": {
      "search": "搜尋表格",
      "lengthMenu": "每頁顯示 _MENU_ 筆",
      "zeroRecords": "無資料",
      "info": "第 _START_ 到 _END_ 筆 共 _TOTAL_ 筆",
      "infoEmpty": "無資料",
      "infoFiltered": "(filtered from _MAX_ total records)",
      "paginate": {
        'first': '第一頁',
        'previous': '前一頁',
        'next': '下一頁',
        'last': '最後一頁',
      }
    }
  });

  // document.getElementById('show_input').innerHTML = JSON.stringify(data['data'][0], undefined, 2);

}

$(function() {
    $(document).ready(function() {
      $('#targetTable').DataTable(
        
        {
          // ajax: 'objects_salary.txt',
          // "oLanguage": {
          //   "sSearch": "搜尋表格",
          //   "sLengthMenu": "每次顯示 _MENU_ 筆 "
            
          // }
          
          "language": {
            "search": "搜尋表格",
            "lengthMenu": "每頁顯示 _MENU_ 筆",
            "zeroRecords": "無資料",
            "info": "第 _START_ 到 _END_ 筆 共 _TOTAL_ 筆",
            "infoEmpty": "無資料",
            "infoFiltered": "(filtered from _MAX_ total records)",
            "paginate": {
              'first': '第一頁',
              'previous': '前一頁',
              'next': '下一頁',
              'last': '最後一頁',
            }
            
          }
        } 
        // {
        //   order: [[4, 'aesc']],
        //   columns: [
        //       null,
        //       null,
        //       null,
        //       null,
        //       null,
        //       null,
        //       { orderable: false},
        //       // { orderable: false, width: "8%"  }
        //   ]
        // }
      );
    });
  });



spans = document.querySelectorAll(".copy-text");
keyword = document.querySelector('.keyword.copy-text');
if (keyword){
  spans = Array.prototype.slice.call(spans, 1, spans.length);
}

if(spans){
  for (var span of spans) {
    span.onclick = function(event) {
      // document.execCommand("copy");
      let copyText = event.target.textContent;
      navigator.clipboard.writeText(copyText);
      // const tmp_toast = document.createElement("div");
      // tmp_toast.innerHTML = `
      // <div class="toast">
      //   <div class="toast-header">
      //     Toast Header
      //   </div>
      //   <div class="toast-body">
      //     Some text inside the toast body
      //   </div>
      // </div>
      // `
      // span.appendChild(tmp_toast);
      
      $('.toast').toast('show');
      // $('#liveToast').toast('show');
      random_keyword = document.querySelector('.query-or-random')
      if(random_keyword){
        if(random_keyword.querySelector('b').innerHTML=="隨機關鍵字"){
          $('#id_keyword').val(copyText)
        }
        else{
          $('#id_opinion').val(copyText)

        }
      }
      else{
        $('#id_opinion').val(copyText)

      }

      
    }
    if (keyword){
      var span_text = span.innerHTML;
      var keyword_text = keyword.innerHTML
      // console.log(keyword_text)
      span.innerHTML = span_text.replace(new RegExp(keyword_text, "g"),"<span style='background-color: yellow;'>"+keyword_text+"</span>");
      // console.log(span.innerHTML)
    }
  }
}

// Datatable 設定參考
// return array(
// 	'emptyTable'     => 'No data available in table',
// 	'info'           => 'Showing _START_ to _END_ of _TOTAL_ entries',
// 	'infoEmpty'      => 'Showing 0 to 0 of 0 entries',
// 	'infoFiltered'   => '(filtered from _MAX_ total entries)',
// 	'infoPostFix'    => '',
// 	'lengthMenu'     => 'Show _MENU_ entries',
// 	'loadingRecords' => 'Loading...',
// 	'processing'     => 'Processing...',
// 	'search'         => 'Search:',
// 	'zeroRecords'    => 'No matching records found',
// 	'paginate'       => array(
// 		'first'    => 'First',
// 		'previous' => 'Previous',
// 		'next'     => 'Next',
// 		'last'     => 'Last',
// 	),
// 	'aria'           => array(
// 		'sortAscending'  => ': activate to sort column ascending',
// 		'sortDescending' => ': activate to sort column descending',
// 	),
// 	'decimal'        => '',
// 	'thousands'      => ',',
// );

// $('#test').DataTable( {
//   columnDefs: [
//     { orderable: false, targets: 6 }
//   ]
// } );

// new DataTable('#example', {
//   dom: '<"toolbar">frtip'
// });

// document.querySelector('div.toolbar').innerHTML = '<b>Custom tool bar! Text/images etc.</b>';

// new DataTable('#example', {
//   dom: '<"top"i>rt<"bottom"flp><"clear">'
// });